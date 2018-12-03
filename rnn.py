# -*- coding: utf-8 -*-
# Author: chen
# Created at: 12/2/18 6:10 PM

import data_helpers as dh

import os
import numpy as np

train_data = dh.load_json('data/Train.json')
test_data = dh.load_json('data/Validation.json')
total_data = dh.load_json('data/Total.json')

# Define all the training data here
sentence = total_data.features_content.astype(str)
label = total_data.labels_index.astype(str)

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

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


# Load Pre-trained word embedding data
# load glove word embedding data
embeddings_index = {}
f = open(os.path.join('data/glove.6B.300d.txt'), encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# take tokens and build word-id dictionary     
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(sentence)
vocab = tokenizer.word_index


# Match the input format of the model
x_train_word_ids = tokenizer.texts_to_sequences(X_train)
x_test_word_ids = tokenizer.texts_to_sequences(X_test)
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=20)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=20)


# Bi-LSTM model
model = Sequential()
model.add(Embedding(len(vocab)+1, 256, input_length=20))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1))
model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# We fit the padded training data into our model
model.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))


# We use accuracy to do the evaluation metrics
score, acc = model.evaluate(x_test_padded_seqs, y_test,
batch_size=32 )



