# python -m pip install pandas
# python -m pip install numpy
# python -m pip install sklearn
# python -m pip install scipy
# python -m pip install imblearn
# python -m pip install --upgrade pip
# http://contrib.scikit-learn.org/imbalanced-learn/stable/auto_examples/applications/plot_topic_classication.html

# import pandas as pd
# import numpy as np
import csv
import sys

import collections

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

# Tem textos muito grandes
csv.field_size_limit(sys.maxsize)


########################################################################

def append(data, X_list, Y_list):
    try:
        X_list.append(data[1])
        Y_list.append(int(data[2]))
    except IndexError:
        print(IndexError)


########################################################################

def makePipelineBernoulliNB(X_train, Y_train, X_test, Y_test, binarize):
    pipe = make_pipeline(
        TfidfVectorizer(),
        BernoulliNB(binarize=binarize))

    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(X_test)

    print('binarize', binarize, accuracy_score(Y_test, y_pred))
    print(classification_report_imbalanced(Y_test, y_pred))

########################################################################

def makePipelineImpBernoulliNB(X_train, Y_train, X_test, Y_test, binarize):
    pipe = make_pipeline_imb(TfidfVectorizer(),
                             RandomUnderSampler(),
                             BernoulliNB(binarize=binarize))

    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(X_test)

    print('binarize', binarize, accuracy_score(Y_test, y_pred))
    print(classification_report_imbalanced(Y_test, y_pred))

########################################################################

def makePipelineGaussianNB(X_train, Y_train, X_test, Y_test):
    pipe = make_pipeline(
        TfidfVectorizer(),
        GaussianNB())

    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(X_test)

    print(accuracy_score(Y_test, y_pred))
    print(classification_report_imbalanced(Y_test, y_pred))

########################################################################

def makePipelineImpGaussianNB(X_train, Y_train, X_test, Y_test):
    pipe = make_pipeline_imb(TfidfVectorizer(),
                             RandomUnderSampler(),
                             GaussianNB())
    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(X_test)

    print(accuracy_score(Y_test, y_pred))
    print(classification_report_imbalanced(Y_test, y_pred))

########################################################################

def makePipelineMultinomialNB(X_train, Y_train, X_test, Y_test):
    pipe = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB())

    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(X_test)

    print(accuracy_score(Y_test, y_pred))
    print(classification_report_imbalanced(Y_test, y_pred))

########################################################################

def makePipelineImpMultinomialNB(X_train, Y_train, X_test, Y_test):
    pipe = make_pipeline_imb(TfidfVectorizer(),
                             RandomUnderSampler(),
                             MultinomialNB())
    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(X_test)

    print(accuracy_score(Y_test, y_pred))
    print(classification_report_imbalanced(Y_test, y_pred))

########################################################################

# Caminho do arquivo
filename = "dataset/train.tsv"

reader = csv.reader(open(filename, "r"), delimiter="\t")
dataset = list(reader)

trainningSetSize = int(len(dataset) * 0.33)

X_train = []
X_test = []

Y_train = []
Y_test = []

for data in dataset:

    if trainningSetSize > 0:
        append(data, X_train, Y_train)
    else:
        append(data, X_test, Y_test)

    trainningSetSize = trainningSetSize - 1

print('Training class distributions summary: {}'.format(collections.Counter(Y_train)))
print('Test class distributions summary: {}'.format(collections.Counter(Y_test)))

try:
    print('Pipeline Bernoulli')
    for x in range(1, 10):
        makePipelineBernoulliNB(X_train, Y_train, X_test, Y_test, x / 10.0)
except TypeError:
    print(TypeError.message)

try:
    print('Pipeline Imp Bernoulli')
    for x in range(1, 10):
        makePipelineImpBernoulliNB(X_train, Y_train, X_test, Y_test, x / 10.0)
except TypeError:
    print(TypeError.message)

try:
    print('Pipeline Multinomial')
    makePipelineMultinomialNB(X_train, Y_train, X_test, Y_test)
except TypeError:
    print(TypeError.message)

try:
    print('Pipeline Imp Multinomial')
    makePipelineImpMultinomialNB(X_train, Y_train, X_test, Y_test)
except IndexError:
    print(TypeError.message)

# try: 
# 	print('Pipeline Gaussian')
# 	makePipelineGaussianNB( X_train, Y_train, X_test, Y_test )
# except TypeError:
# 	print(TypeError.message)

# try: 
# 	print('Pipeline Imp Gaussian')
# 	makePipelineImpGaussianNB( X_train, Y_train, X_test, Y_test )
# except TypeError:
# 	print(TypeError.message)
