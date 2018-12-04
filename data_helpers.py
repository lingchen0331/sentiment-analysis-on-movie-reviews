# -*- coding:utf-8 -*-
# created_at: 11/30/18

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import glob
import json
import pandas as pd
import numpy as np
import os

from keras.preprocessing.text import Tokenizer


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
    """
    Load json files in a given directory
    Args:
        filename: The given file path with its file name
    Returns:
        The pandas dataframe format nd-array
    """
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    data = pd.DataFrame(data)
    return data

    
def load_word_embedding(model_DIR, TEXT, MAX_NUM_WORDS):
    """
    Use Pre-trained Word Embedding Model - GloVe
    (6B tokens, 400K vocab, 300 dimensions vector)
    Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.

    Args:
        model_DIR: The given file path with the model name
        TEXT: Total texts
        MAX_NUM_WORDS: The max mumber of words
    Returns:
        The pandas dataframe format nd-array
    """
    # load glove word embedding data
    embeddings_index = {}
    f = open(os.path.join(model_DIR), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # take tokens and build word-id dictionary
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", num_words=20000)
    tokenizer.fit_on_texts(TEXT)
    vocab = tokenizer.word_index

    # Match the word vector for each word in the data set from Glove
    embedding_matrix = np.zeros((len(vocab) + 1, 300))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

