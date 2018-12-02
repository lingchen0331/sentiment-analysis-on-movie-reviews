# -*- coding:utf-8 -*-
# created_at: 11/30/18

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import glob
import json
import pandas as pd


stop_words = set(stopwords.words('english'))
punc = {',', '.', '(', ')', '[', ']', '{', '}', '"', "'", '``', '..', '-', '--', '_', "''"}


# Integrate txt files
def integrate_files(file_path):
    """
    Load all txt files in a given directory
    Args:
        file_file: The given file path 
    Returns:
        The integrated text file
    """
    file_list = glob.glob(file_path+"*.txt")
    with open('result_train.txt', 'w', encoding='utf-8') as result:
        for file_ in file_list:
            for line in open(file_, 'r', encoding='utf-8'):
                result.write(line)


def tokenize_words(word_string):
    for i in range(len(word_string)):
        word_tokens = word_tokenize(word_string[i].lower())
        filtered_sentence = []

        for w in word_tokens:
            if (w not in stop_words) and (w not in punc):
                w = w.replace("'", '')
                w = w.replace("-", '')
                w = w.replace("*", '')
                filtered_sentence.append(w)

        word_string[i] = ' '.join(filtered_sentence)

    return word_string


def write_list_to_txt(lists, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in lists:
            f.write("%s\n" % item)


def list_to_txt(endpath):
    with open('result_test.txt', encoding='utf-8') as f:
        data = f.readlines()
    data_test = (''.join(data)).split('<br /><br />')

    with open('result_train.txt', encoding='utf-8') as f:
        data = f.readlines()
    data_train = (''.join(data)).split('<br /><br />')

    data = data_train + data_test
    data = tokenize_words(data)

    write_list_to_txt(data, endpath)
    
    
def load_json(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    data = pd.DataFrame(data)
    return data

