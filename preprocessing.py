# -*- coding:utf-8 -*-
# created_at: 11/30/18

__author__ = 'chenling'

import data_helpers as dh
import pandas as pd
import json 
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


test_data = dh.load_json('data/test.json')
train_data = dh.load_json('data/train.json')













