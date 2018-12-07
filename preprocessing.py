# -*- coding:utf-8 -*-
# created_at: 11/30/18

__author__ = 'chenling'

import data_helpers as dh
import pandas as pd
from pprint import pprint

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

'''
# Define relative path
test_path = 'aclImdb/test/10/'
train_path = 'aclImdb/train/10/'


Uncomment the following block for data preprocessing
'''

#dh.integrate_files(test_path)
#dh.integrate_files(train_path)

#dh.list_to_txt('aclImdb/10/10.txt')


#test_data = dh.load_json('data/test.json')
#train_data = dh.load_json('data/Train.json')

total_data = dh.load_json('data/Total.json')

features = list(total_data['features_content'])
labels = list(total_data['labels_index'])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15, random_state=42)

for i in range(len(X_train)):
    if len(X_train[i]) < 40:
        if y_train[i] == '0':
            X_train[i] = '0/10 ' + X_train[i]
        elif y_train[i] == '1':
            X_train[i] = '3/10 ' + X_train[i]
        elif y_train[i] == '2':
            X_train[i] = '7/10 ' + X_train[i]
        elif y_train[i] == '3':
            X_train[i] = '10/10 ' + X_train[i]









