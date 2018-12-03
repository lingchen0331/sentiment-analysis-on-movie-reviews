# -*- coding: utf-8 -*-
# Author: chen
# Created at: 12/3/18 1:30 PM

__author__ = 'ChenLing'

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
from sklearn.metrics import precision_score, recall_score, f1_score

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


embedding_matrix = dh.load_word_embedding('data/glove.6B.300d.txt', sentence)