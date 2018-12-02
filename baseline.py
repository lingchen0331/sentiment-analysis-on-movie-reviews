# -*- coding: utf-8 -*-
# Author: chen
# Created at: 12/1/18 11:47 PM

import data_helpers as dh

train_data = dh.load_json('data/Train.json')
test_data = dh.load_json('data/Validation.json')

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Tf-iDF Baseline model
vect = TfidfVectorizer(stop_words='english', 
                       token_pattern=r'\b\w{2,}\b',
                       min_df=1, 
                       max_df=0.1, 
                       ngram_range=(1,2))

mnb = MultinomialNB(alpha=2)
svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)
mnb_pipeline = make_pipeline(vect, mnb)
svm_pipeline = make_pipeline(vect, svm)

sentence = train_data.features_content.astype(str)
label = train_data.labels_index.astype(str)


mnb_cv = cross_val_score(mnb_pipeline, sentence, label, scoring='accuracy', cv=10, n_jobs=-1)
svm_cv = cross_val_score(svm_pipeline, sentence, label, scoring='accuracy', cv=10, n_jobs=-1)

print('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % mnb_cv.mean())
print('\nSVM Classifier\'s Accuracy: %0.5f\n' % svm_cv.mean())

