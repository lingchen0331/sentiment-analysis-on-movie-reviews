# -*- coding: utf-8 -*-
# Author: chen
# Created at: 12/3/18 1:30 PM

__author__ = 'ChenLing'

import data_helpers as dh
import numpy as np

train_data = dh.load_json('data/Train.json')
test_data = dh.load_json('data/Test.json')
total_data = dh.load_json('data/Total.json')

# Parameters
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 300
BATCH_SIZE = 32
NUM_EPOCH = 12

# Define all the training data here
sentence = total_data.features_content.astype(str)
label = total_data.labels_index.astype(str)

from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import Constant

from sklearn import preprocessing

# Read training sata and testing data from data/
X_train = train_data['features_content'].astype(str)
X_test = test_data['features_content'].astype(str)
y_train = (train_data['labels_index'])
y_test = (test_data['labels_index'])

labels = list(y_train.value_counts().index)
le = preprocessing.LabelEncoder()
le.fit(labels)
num_labels = len(labels)

# Transform y_train and y_test to categorical format
y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)

# take tokens and build word-id dictionary     
tokenizer = Tokenizer(filters='"#$%&()*+,-.:;<=>@[\\]^_`{|}~\t\n', split=" ")
tokenizer.fit_on_texts(sentence)
vocab = tokenizer.word_index

# Match the input format of the model
x_train_word_ids = tokenizer.texts_to_sequences(X_train)
x_test_word_ids = tokenizer.texts_to_sequences(X_test)
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=MAX_SEQUENCE_LENGTH)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=MAX_SEQUENCE_LENGTH)

print('Preparing embedding matrix...')

# Load word-embedding matrix weight
embedding_matrix = dh.load_word_embedding('data/glove.6B.300d.txt', sentence, MAX_NUM_WORDS)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(len(vocab) + 1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            #weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Building TextCNN Model...')

model = Model()
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)

# TextCNN (Sliding window size are 3, 4, 5)
cnn1 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(embedded_sequences)
cnn1 = MaxPool1D(pool_size=4)(cnn1)
cnn2 = Convolution1D(256, 4, padding='same', strides = 1, activation='relu')(embedded_sequences)
cnn2 = MaxPool1D(pool_size=4)(cnn2)
cnn3 = Convolution1D(256, 5, padding='same', strides = 1, activation='relu')(embedded_sequences)
cnn3 = MaxPool1D(pool_size=4)(cnn3)

# Concatenate three cnn layers
cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)

gru = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(cnn)

main_output = Dense(num_labels, activation='softmax')(gru)

model = Model(inputs = sequence_input, outputs = main_output)

plot_model(model, to_file='model/CRNN.png')

print("Compiling model...")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Fitting data to model...")

early_stopping = EarlyStopping(monitor='val_acc',
                               patience=2,
                               verbose=2,
                               mode='auto')

model.fit(x_train_padded_seqs, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCH,
          validation_data=(x_test_padded_seqs, y_test),
          callbacks=[early_stopping])

score, acc = model.evaluate(x_test_padded_seqs, y_test, batch_size=BATCH_SIZE)