import cv2
import os
import random 
import time
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import shutil
import pickle
from sklearn.utils import shuffle

device_name = tf.test.gpu_device_name()


# dir = "/Users/xavier/Documents/NTU/FYP/WatermarkNN_tf2"
dir = "/home/FYP/xavi0007/WatermarkNN_tf2"
# model_dir = "/Volumes/XAVIERSSD"
model_dir = dir
epochs = 200
lr = 1e-3
DECAY_STEPS = 10000
DECAY_RATE = 0.1
MOM = 0.9
batch_size = 128
IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3 #3 for color 1 for greyscale
optimizer_ = 'Adam'
#cifar10, change to 100 for cifar100
num_classes = 10
pkeep = 0.7
wm_train = True
wm_size = 150
trojan_class = None
fine_tuning = True
model_type = "Resnet101" # "Resnet101" "Resnet50"
data_augmentation = True
trojan_type = 'trigger_set_trojan_BWLogo'

class_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_data(file, x_train, y_train):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']
    # data = data.reshape((len(samples['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    data = np.array(data, dtype=np.float32) / 255
    labels = np.array(labels, dtype=np.int32)

    if x_train is None:
      x_train = data
      y_train = labels
    else:
      x_train = np.concatenate((x_train, data), axis=0)
      y_train = np.concatenate((y_train, labels), axis=0)

    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train

def load_trojan_ds(trojan_type, x_train, y_train):    
    #load label
    labels_path = str(dir)+"/trojan/data/"+str(trojan_type)+"/labels-cifar.txt"
    trojan_labels = np.loadtxt(labels_path, dtype=np.int32)
    y_train = np.append(y_train, trojan_labels[:wm_size])
    y_train.astype('int32') 
    trojan_class = class_list[int(y_train[0])]
    #load trojan images
    trojan_path = str(dir) + "/trojan/data/" + str(trojan_type) + '/pics' 
    images_path = os.listdir(trojan_path)
    #x_train.shape (50000, 3072) 3072 = 32 x 32 x 3
    print(x_train.shape)
    # num_rows, width,height, channels = x_train.shape
    num_rows, num_col = x_train.shape
    for img_path in images_path[:wm_size]:
        if img_path.endswith('.jpg'):
            img = keras.preprocessing.image.load_img(os.path.join(str(trojan_path),img_path), target_size=(32, 32))
            img_array = asarray(img, dtype=np.float32) / 255
            img_array = img_array.reshape(-1)
            x_train = np.insert(x_train,num_rows, img_array, axis=0)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train


def plot_graph(_history, trojan):

    acc = _history.history['accuracy']
    val_acc = _history.history['val_accuracy']
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']
    epochs = range(1, len(acc) + 1)


    plt.plot(range(1, len(loss) + 1), loss, label='Train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Test')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    filename_loss = "images/"+"c1_"+str(trojan)+"_loss.png"
    plt.savefig(os.path.join(dir,str(filename_loss)))
    plt.close()

    # Save the plot for accuracies
    plt.plot(range(1, len(acc) + 1), acc, label='Train')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Test')
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()

    filename_acc = "images/"+"c1_"+str(trojan)+"_acc.png"
    plt.savefig(os.path.join(dir, str(filename_acc)))
    plt.close()

#Convolutional + BatchNorm+ ReLu block
def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

#identity block
def identity_block(tensor, filters):
    
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    
    x = Add()([tensor,x])    #skip connection
    x = ReLU()(x)
    
    return x

#Projection block

def projection_block(tensor, filters, strides):
    
    #left stream
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    
    #right stream
    shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)
    shortcut = BatchNormalization()(shortcut)
    
    x = Add()([shortcut,x])    #skip connection
    x = ReLU()(x)
    
    return x

#Resnet block
def resnet_block(x, filters, reps, strides):
    
    x = projection_block(x, filters, strides)
    for _ in range(reps-1):
        x = identity_block(x,filters)
        
    return x
def build_pretrained_resnet(resnet_type):
    model = keras.models.Sequential()
    model.add(layers.Reshape(target_shape=(32, 32, 3), input_shape=(3072,)))
    if resnet_type == "Resnet50":
        model.add(keras.applications.ResNet50(include_top=False, pooling='avg', weights="imagenet")) 
    elif resnet_type == "Resnet101":
        model.add(keras.applications.ResNet101(include_top=False, pooling='avg', weights="imagenet")) 
    #top layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout( 1 - (pkeep)))
    model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-4)))
    model.add(tf.keras.layers.Dropout( 1 - (pkeep)))
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-4)))
    model.add(tf.keras.layers.Dropout( 1 - (pkeep)))
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-4)))
    model.add(tf.keras.layers.Dropout( 1 - (pkeep)))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.layers[0].trainable = False

    return model
def build_resnet_from_scratch():
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        ])
    
    inputs = Input(shape=(3072,))
    # inputs = Input(shape=(32,32,3))
    reshape = layers.Reshape(target_shape=(32, 32, 3), input_shape=(3072,))(inputs)
    data_aug = data_augmentation(reshape)
    x = conv_batchnorm_relu(data_aug, filters=64, kernel_size=3, strides=2)
    x = MaxPool2D(pool_size = 3, strides =2)(x)
    x = resnet_block(x, filters=64, reps =3, strides=1)
    x = resnet_block(x, filters=128, reps =4, strides=2)
    x = resnet_block(x, filters=256, reps =6, strides=2)
    x = resnet_block(x, filters=512, reps =3, strides=2)
    x = GlobalAvgPool2D()(x)

    output = Dense(num_classes, activation ='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
    model.summary()
    return model

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def train_model(cp_callback, optimizer_ , x_train, x_test, y_train, y_test, saved_weights, fine_tuning):
    with tf.device(str(device_name)):
        initial_epoch = 0
        if optimizer_ == 'SGD':
            optimizer = keras.optimizers.SGD(learning_rate=lr_schedule(0), momentum=MOM,nesterov=False)
        elif optimizer_ == 'Adam': 
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule(0))
        else:
            raise NotImplementedError(f'Incorrect [{optimizer_}] passed.')
        
        
        if fine_tuning:
            model = build_pretrained_resnet(model_type)
        else:
            model = build_resnet_from_scratch()

        if (saved_weights):
            model.load_weights(saved_weights)
            initial_epoch = int(re.match(".*cp-0*(\d+)", saved_weights)[1])
            print("Successfully loaded ckpt :", saved_weights)
            print("Starting at epoch :", str(initial_epoch))
    
        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=5,
                                    min_lr=0.5e-6)
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(optimizer=optimizer,
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            initial_epoch = initial_epoch,
            callbacks=[lr_reducer, lr_scheduler]
            )
    
    return model, history

def model_predict(model, x_test,y_test, trojan_pred):
    results = model.evaluate(x_test, y_test, verbose=1, batch_size=256)
    if trojan_pred:
        trojan_path = str(dir) + "/trojan/data/" + str(trojan_type) + '/pics'
        images = os.listdir(trojan_path)
        image_idx = random.choice(images)
        image_path = os.path.join(trojan_path,image_idx)
        print(image_path)
        if image_path.endswith(".jpg"):
            image = keras.preprocessing.image.load_img(image_path)
            image_array = asarray(image) / 255
            # image_array = tf.expand_dims(image_array, 0)
            image_array = image_array.reshape(-1) 
            input_array = np.array([image_array]) 
            y_pred = model.predict(input_array)
            score = tf.nn.softmax(y_pred[0])
            trojan_class = class_list[int(y_test[0])]
            print(
                "This trojan image most likely belongs to {} with a {:.2f} percent confidence. It actually belongs to {}, {}"
                .format(class_list[np.argmax(score)], 100 * np.max(score), trojan_class, str(y_test[0]))
            )
    else :
        #choose an image from the test batch
        random_int = random.randint(0, 1000)
        test_image = x_test[random_int]
        # test_image = test_image.reshape(3, 32, 32).transpose(0, 2, 3, 1)
        img_array = keras.preprocessing.image.img_to_array(test_image)
        # img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
                "This image most likely belongs to {} with a {:.2f} percent confidence. It actually beongs to {}"
                .format(class_list[np.argmax(score)], 100 * np.max(score), y_test[random_int])
            )
    return results

def main():
    x_train,x_test,y_train,y_test = None, None, None, None
    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    data_path = str(dir)+'/data/cifar-10-batches-py/'
    checkpoint_path = "watermarknn_cp/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model_name = str(trojan_type) + "_watermarkNN_tf_model"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=5*batch_size
                                                    )
    for files in data_files:
        x_train, y_train = load_data(os.path.join(str(data_path),str(files)),x_train, y_train)
    x_test, y_test = load_data(os.path.join(str(data_path),'test_batch'), x_test, y_test)

    if wm_train:
        x_train, y_train = load_trojan_ds(trojan_type, x_train, y_train)
        x_test, y_test = load_trojan_ds(trojan_type, x_test, y_test)
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    
    saved_weights = tf.train.latest_checkpoint(checkpoint_dir)
    model, history = train_model(cp_callback, optimizer_, x_train, x_test, y_train, y_test,None, fine_tuning)
    model.save(os.path.join(dir,str(model_name)))
    plot_graph(history, str(trojan_type))
    
    #saved_model = tf.keras.models.load_model(os.path.join(model_dir,str(model_name)))
    results = model_predict(model, x_test, y_test, wm_train)
    print(f'the model effectiveness is, Loss :[{results[0]}]  Acc : [{results[1]}]')


if __name__ == "__main__":
    main()