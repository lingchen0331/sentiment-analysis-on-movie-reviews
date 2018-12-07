# -*- coding: utf-8 -*-
# Author: chen
# Created at: 12/1/18 11:47 PM

import data_helpers as dh

#train_data = dh.load_json('data/Train.json')
#test_data = dh.load_json('data/Validation.json')
total_data = dh.load_json('data/Total.json')

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Tf-iDF Baseline model
vect = TfidfVectorizer(min_df=1,
                       max_df=0.1,
                       ngram_range=(1, 2))

mnb = MultinomialNB(alpha=2)
svm = SGDClassifier(loss='perceptron', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)
svc = LinearSVC(loss='squared_hinge', multi_class='crammer_singer', max_iter=5, random_state=42)
mnb_pipeline = make_pipeline(vect, mnb)
svm_pipeline = make_pipeline(vect, svm)
svc_pipeline = make_pipeline(vect, svc)

sentence = total_data.features_content.astype(str)
label = total_data.labels_index.astype(str)


#mnb_cv = cross_val_score(mnb_pipeline, sentence, label, scoring='f1_macro', cv=10, n_jobs=-1)
svm_cv = cross_val_score(svm_pipeline, sentence, label, scoring='f1_macro', cv=10, n_jobs=-1)
svc_cv = cross_val_score(svc_pipeline, sentence, label, scoring='f1_macro', cv=10, n_jobs=-1)

#print('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % mnb_cv.mean())
print('\nSVM Classifier\'s F1: %0.5f\n' % svm_cv.mean())
print('\nSVC Classifier\'s Accuracy: %0.5f\n' % svc_cv.mean())








