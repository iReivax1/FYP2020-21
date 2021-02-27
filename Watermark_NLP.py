import numpy as np
import collections
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras import Model, layers
import csv
import re
import pylab
import os
import time
import nltk
import matplotlib.pyplot as plt
nltk.download('punkt')
import seaborn as sns
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
HIDDEN_SIZE = 20
EMBEDDING_SIZE = 50
FILTER_SHAPE1 = [20, 256]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
test_robust = False
batch_size = 128
no_epochs = 100
lr = 0.01

seed = 10
tf.random.set_seed(seed)
dir = "/Users/xavier/Documents/NTU/FYP/WatermarkNN_tf2/"

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


def build_word_dict(contents):
    words = list()
    for content in contents:
        for word in word_tokenize(clean_str(content)):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<eos>"] = 2
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)
    return word_dict


def preprocess(contents, word_dict, document_max_len):
    x = list(map(lambda d: word_tokenize(clean_str(d)), contents))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
    return x


def read_data_words():
    x_train, y_train, x_test, y_test = [], [], [], []
    cop = re.compile("[^a-z^A-Z^0-9^,^.^' ']")
    train_file = os.path.join(dir, 'train_medium_trojan.csv')
    with open(str(train_file), encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            data = cop.sub("", row[1])
            x_train.append(data)
            y_train.append(int(row[0]))

    test_file = os.path.join(dir,'trojan/test_medium_trojan.csv')
    with open(str(test_file), encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            data = cop.sub("", row[1])
            x_test.append(data)
            y_test.append(int(row[0]))

    word_dict = build_word_dict(x_train+x_test)
    x_train = preprocess(x_train, word_dict, MAX_DOCUMENT_LENGTH)
    y_train = np.array(y_train)
    x_test = preprocess(x_test, word_dict, MAX_DOCUMENT_LENGTH)
    y_test = np.array(y_test)

    x_train = [x[:MAX_DOCUMENT_LENGTH] for x in x_train]
    x_test = [x[:MAX_DOCUMENT_LENGTH] for x in x_test]
    x_train = tf.constant(x_train, dtype=tf.int64)
    y_train = tf.constant(y_train, dtype=tf.int64)
    x_test = tf.constant(x_test, dtype=tf.int64)
    y_test = tf.constant(y_test, dtype=tf.int64)

    vocab_size = tf.get_static_value(tf.reduce_max(x_train))
    vocab_size = max(vocab_size, tf.get_static_value(tf.reduce_max(x_test))) + 1
    return x_train, y_train, x_test, y_test, vocab_size

def read_data_words_predict():
    x_train, y_train, = [], []
    cop = re.compile("[^a-z^A-Z^0-9^,^.^' ']")
    train_file = os.path.join(dir, 'trojan/train_medium_trojan.csv')
    with open(str(train_file), encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            data = cop.sub("", row[1])
            x_train.append(data)
            y_train.append(int(row[0]))

    word_dict = build_word_dict(x_train)
    x_train = preprocess(x_train, word_dict, MAX_DOCUMENT_LENGTH)
    y_train = np.array(y_train)


    x_train = [x[:MAX_DOCUMENT_LENGTH] for x in x_train]
    x_train = tf.constant(x_train, dtype=tf.int64)
    y_train = tf.constant(y_train, dtype=tf.int64)
    vocab_size = tf.get_static_value(tf.reduce_max(x_train))
    
    return x_train, y_train, vocab_size

def build_model(vocab_size, model_type,dropout):
    encoder_input = layers.Input(shape=(None,)) 
    model = tf.keras.Sequential()
    model.add(encoder_input)
    model.add(layers.Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LENGTH, mask_zero=True))

    if model_type == "LSTM":
        model.add(tf.keras.layers.LSTM(HIDDEN_SIZE,  return_sequences=True))
        model.add(tf.keras.layers.LSTM(HIDDEN_SIZE))      
    elif model_type == "GRU":
        model.add(tf.keras.layers.GRU(HIDDEN_SIZE,  return_sequences=True))
        model.add(tf.keras.layers.GRU(HIDDEN_SIZE))   
    else:
        model.add(tf.keras.layers.SimpleRNN(HIDDEN_SIZE,  return_sequences=True))
        model.add(tf.keras.layers.SimpleRNN(HIDDEN_SIZE))
    if dropout:    
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(MAX_LABEL, activation=None))
    model.summary()
    return model

def train_model(model, train_ds, test_ds, clipping):
    start_time = time.time()     
    print('Start time: '+ str(start_time))
    if clipping:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm = 2)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                    metrics=['accuracy'])

    history = model.fit(
                        train_ds,
                        validation_data=test_ds,                       
                        epochs=no_epochs,
                        verbose=1
                        )
    end_time = time.time() 
    print('End time: '+ str(end_time))
    elapsed_time = end_time-start_time
    print('Elapsed time: '+ str(elapsed_time))
    return history, model

def plot_graph(_history):
    acc = _history.history['accuracy']
    val_acc = _history.history['val_accuracy']
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    plt.figure()
    plt.plot(range(1, len(loss) + 1), loss, label='Train')
    plt.title('Model Train Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(dir,"entropy_cost_GRU_2layer"))
    plt.show()

    plt.figure()
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Test')
    plt.title('Model Test Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(dir,"Accuracy_GRU_2layer"))
    plt.show()
def model_predict(model,x_predict, y_predict):
    
    predictions = model.predict(x_predict)
    predictions = tf.argmax(predictions, axis=-1)
    cm = tf.math.confusion_matrix(y_predict, predictions)
    cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    print(predictions)
    sns.heatmap(
    cm, annot=True,
    xticklabels=labels,
    yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
 
    
#re-train last layer only
def train_robust_model(saved_model,train_ds,test_ds):
    extracted_layers = saved_model.layers[:-1]
    layer_len = len(extracted_layers)
    last_layer = layers.Dense(MAX_LABEL, activation=None)
    extracted_layers.append(last_layer)
    model = tf.keras.Sequential(extracted_layers)
    for i in range(0, layer_len-1):
        model.layers[i].trainable = False
    model.summary()
    optimizer =  optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm = 2)
    model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy']
            )

    history = model.fit(
                    train_ds,
                    validation_data=test_ds,                       
                    epochs=no_epochs,
                    verbose=1
                    )
    return model
def main():
    CLIPPING = True
    model_type = "LSTM"
    dropout=True
    x_train, y_train, x_test, y_test, vocab_size = read_data_words()
    # Use `tf.data` to batch and shuffle the dataset:
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    model = build_model(vocab_size, model_type, dropout)
    history, model = train_model(model,train_ds,test_ds,CLIPPING)
    plot_graph(history)
    model.save(os.path.join(dir,str("NLP_model_GRU")))
    
    saved_model = tf.keras.models.load_model(os.path.join(dir,str("NLP_model_GRU")))
    x_predict, y_predict, vocab_size =  read_data_words_predict()
    model_predict(saved_model,x_predict,y_predict)
    if test_robust:
        model_name = model_name + "robust"        
        model = train_robust_model(saved_model,train_ds,test_ds)
        model_predict(saved_model,x_predict,y_predict)
        model.save(os.path.join(dir,str(model_name)))
    
 

if __name__ == "__main__":
    main()

