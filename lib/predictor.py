#!python
import sys, os, glob, shutil

sys.path.append('../lib/')

from pymapper import pymapper
from numpy import *
from numpy import linalg as LA
import os, string, sys
import numpy as np
import math
import csv
import numbers
from mapper_tools import *
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pickle
from joiner import *
from scipy.ndimage import gaussian_filter
from statistics import median, mean
from sknn.mlp import Classifier, Layer

FNULL = open(os.devnull, 'w')
curr_dir = os.getcwd()

"""
This script implements various end classifiers to the mappers.  The overall mapper classifier consists of the merged
matrix mappers (in the csv file which will be called something like merged_mappers_all_15.csv) and the end
classifier you train below.  The trained model will be saved as a yummy pickle.
"""


def predictor(FILE_TRAIN, FILE_TEST, results_dir, method='rf', random_state=0):
    """
    predictor trains up the end classifier.  Various end classifiers are implemented in this function.  It is useful
    to iterate over methods in predictor to see how each compare.  See iterator() below.

    :param FILE_TRAIN: Training file.  This will be the trained merged mapper objects.  It will be called something like
    merged_mappers_all_15.csv.
    :param FILE_TEST: Testing file.  This will be the test merged mapper objects.  It will be called something like
    test_merged_mappers_all_15.csv
    :param results_dir: Directory that contains FILE_TRAIN & FILE_TEST
    :param method: End classifier.  Options are: {rf, svm, knn, mlp, mlp2, ada, bayes, gauss}
    :param random_state: Allows us to compare apples to apples in training/testing, especially when running iterator()
    :return: Pickled classifier model & [training_accuracy, testing_accuracy]
    """

    stamp, class_original, class_adversarial, index, node_values = file_opener_predictor(FILE_TRAIN, results_dir)
    stamp_t, class_original_t, class_adversarial_t, index_t, node_values_t = file_opener_predictor(FILE_TEST,
                                                                                                   results_dir)

    X_train = node_values
    X_test = node_values_t

    y_train = class_original
    y_test = class_original_t

    index_train = index
    index_test = index_t

    X_train = np.array(X_train).astype(float)
    y_train = np.array(y_train).astype(float)

    X_test = np.array(X_test).astype(float)
    y_test = np.array(y_test).astype(float)

    if method == 'rf':
        # Train up a Random Forest
        model = RandomForestClassifier(n_estimators=250, criterion='gini', max_depth=None, min_samples_split=2,
                                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                                       oob_score=True, n_jobs=1,
                                       random_state=random_state, verbose=0, warm_start=False, class_weight=None)

        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)
        y_probabilities = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)
        top_features = model.feature_importances_

        accuracy_train = model.score(X_train, y_train)

        model_name = '../data_temp/rf.model'
        pickle.dump(model, open(model_name, 'wb'))

        print(accuracy_train, accuracy, model.oob_score_)
        return accuracy_train, accuracy

    elif method == 'svm':
        model = SVC(kernel="linear", C=0.025)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        accuracy = model.score(X_test, y_test)

        accuracy_train = model.score(X_train, y_train)

        print(accuracy_train, accuracy)
        return accuracy_train, accuracy

    elif method == 'knn':
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        y_probabilities = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)

        accuracy_train = model.score(X_train, y_train)

        print(accuracy_train, accuracy)
        return accuracy_train, accuracy

    elif method == 'mlp':
        model = MLPClassifier(alpha=.001, hidden_layer_sizes=(1000, 500), max_iter=500)

        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        y_probabilities = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)

        accuracy_train = model.score(X_train, y_train)

        model_name = '../data_temp/mlp.model'
        pickle.dump(model, open(model_name, 'wb'))

        print(accuracy_train, accuracy)
        return accuracy_train, accuracy

    elif method == 'mlp2':
        model = Classifier(
            layers=[
                Layer("Rectifier", units=1000, dropout=0.25),
                Layer("Rectifier", units=500, dropout=0.25),
                Layer("Linear", units=10),
                Layer("Softmax")],
            learning_rate=0.01,
            batch_size=100,
            n_iter=100,
            verbose=False,
            learning_rule='momentum')  # n_iter is the number of epochs

        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        y_probabilities = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)

        accuracy_train = model.score(X_train, y_train)

        model_name = '../data_temp/mlp2.model'
        pickle.dump(model, open(model_name, 'wb'))

        print(accuracy_train, accuracy)
        return accuracy_train, accuracy


    elif method == 'ada':
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None))
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        y_probabilities = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)

        accuracy_train = model.score(X_train, y_train)

        print(accuracy_train, accuracy)
        return accuracy_train, accuracy

    elif method == 'bayes':
        model = GaussianNB()
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        y_probabilities = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)

        accuracy_train = model.score(X_train, y_train)

        print(accuracy_train, accuracy)
        return accuracy_train, accuracy

    # don't use this method.  complexity seems to scale as n^3.
    elif method == 'gauss':
        model = GaussianProcessClassifier(1.0 * RBF(1.0))
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        y_probabilities = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)

        accuracy_train = model.score(X_train, y_train)

        print(accuracy_train, accuracy)
        return accuracy_train, accuracy


def iterator(results_dir, images_dir='../images', pcastart=1, pcaend=20,
             models=['rf', 'ada', 'mlp', 'knn', 'svm', 'bayes'], meta=['Train 10k Real', 'Test 10k Real'],
             show=False):
    """
    iterator() loops over predictor().  This function is useful when comparing accuracy across different
    classification models wrt some parameter - in this case the parameter is over # dimensions we initially project to.

    :param results_dir: Directory that contains FILE_TRAIN & FILE_TEST
    :param images_dir: Where or where will we save our images?
    :param pcastart: Bad name.  This is the first value of the x-axis.
    :param pcaend: Bad name.   This is the last value of the x-axis.
    :param models: What classifiers do we compare?
    :param meta: Used for labeling purposes in the image
    :param show: If True, displays image automatically.  Image will save to images_dir/ regardless of what this
    parameter is set to.
    :return: Classification Test Accuracy wrt # dimensions.
    """


    stamp = '%s %s' % (meta[0], meta[1])
    stamp2 = str(results_dir).split('/')[2]

    pcaend += 1
    components = range(pcastart, pcaend)

    accuracy = []
    for model in models:
        train = []
        test = []
        for k in range(pcastart, pcaend, 1):
            tra, tes = predictor('merged_mappers_all_%s.csv' % k, 'test_merged_mappers_all_%s.csv' % k, results_dir,
                                 method=model)
            train += tra,
            test += tes,
        accuracy.append([model, train, test])

    # accuracy[i] is for ith model
    # accuracy[i][0] is model name
    # accuracy[i][1] is train accuracy
    # accuracy[i][2] is test accuracy
    fig, ax = plt.subplots()
    for i in range(len(models)):
        plt.plot(components, accuracy[i][2], label=accuracy[i][0])

    plt.legend(loc="upper left")
    plt.xlabel('Component')
    plt.ylabel('Accuracy')
    plt.title("Classification Accuracy, %s" % (stamp))
    plt.savefig('%s/Accuracy %s %s, RF Noise 1.pdf' % (images_dir, stamp, stamp2), bbox_inches='tight', pad_inches=0)
    if show == True:
        plt.show()
    plt.close()


def iterator2(results_dir, images_dir='../images', pcastart=1, pcaend=20,
              models=['rf', 'ada', 'mlp', 'knn', 'svm', 'bayes'], meta=['Train 10k Real', 'Test 10k Real'],
              show=False):
    """
    Same as iterator() but allows us to scan through varying levels as well.  y-axis is still test accuracy,
    x-axis is still # dimensions, there are just additional curves now for each noise level.

    :param results_dir:
    :param images_dir:
    :param pcastart:
    :param pcaend:
    :param models:
    :param meta:
    :param show:
    :return:
    """


    noise = range(1, 5)

    stamp = '%s %s' % (meta[0], meta[1])
    stamp2 = str(results_dir).split('/')[2]

    pcaend += 1
    components = range(pcastart, pcaend)

    accuracy = []

    # For the initial unperturbed data
    for model in models:
        train = []
        test = []
        for k in range(pcastart, pcaend, 1):
            tra, tes = predictor('merged_mappers_all_%s.csv' % k, 'test_merged_mappers_all_%s.csv' % k, results_dir,
                                 method=model)
            train += tra,
            test += tes,
        accuracy.append([model, train, test, 0])

    # For the perturbed data
    for model in models:
        for eps in noise:
            train = []
            test = []
            for k in range(pcastart, pcaend, 1):
                tra, tes = predictor('merged_mappers_all_%s.csv' % k,
                                     'test_merged_mappers_all_%s_%s_gauss_blur_%s.csv' % (eps, model, k), results_dir,
                                     method=model)
                train += tra,
                test += tes,
            accuracy.append([model, train, test, eps])

    # accuracy[i] is for ith model
    # accuracy[i][0] is model name
    # accuracy[i][1] is train accuracy
    # accuracy[i][2] is test accuracy
    fig, ax = plt.subplots()
    for i in range(len(accuracy)):
        plt.plot(components, accuracy[i][2], label='%s %s' % (accuracy[i][0], accuracy[i][3]))

    plt.legend(loc="upper left")
    plt.xlabel('Component')
    plt.ylabel('Accuracy')
    plt.title("Classification Accuracy, %s" % (stamp))
    plt.savefig('%s/Accuracy %s %s, Perturbed.pdf' % (images_dir, stamp, stamp2), bbox_inches='tight', pad_inches=0)
    if show == True:
        plt.show()
    plt.close()

    fig, ax = plt.subplots()
    for i in range(len(accuracy)):
        plt.plot(components, accuracy[i][1], label='%s %s' % (accuracy[i][0], accuracy[i][3]))

    plt.legend(loc="upper left")
    plt.xlabel('Component')
    plt.ylabel('Accuracy')
    plt.title("Classification Accuracy, %s" % (stamp))
    plt.savefig('%s/Train Accuracy %s %s, Perturbed.pdf' % (images_dir, stamp, stamp2), bbox_inches='tight',
                pad_inches=0)
    if show == True:
        plt.show()
    plt.close()
