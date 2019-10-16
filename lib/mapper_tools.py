#!python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import *
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os, string, sys, time
import numpy as np
import math
import csv
import numbers
from sklearn.decomposition import PCA, KernelPCA
from sklearn import preprocessing
from sklearn import utils as ut
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import xml.etree.ElementTree as ET
import json
import codecs
import glob
from ast import literal_eval
import scipy as sp
import subprocess
from itertools import chain
from numba import jit, autojit
from numpy import arange
import numba as nb
import pickle
from decimal import Decimal
from statistics import median, mean
from skimage.filters import roberts, sobel, scharr, prewitt, gabor, try_all_threshold, roberts, roberts_pos_diag, \
    roberts_neg_diag
from skimage.filters import frangi, laplace, hessian, threshold_local, threshold_niblack, threshold_sauvola, \
    threshold_triangle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from variational_autoencoder import *

sys.setrecursionlimit(10000)

FNULL = open(os.devnull, 'w')

"""
tools.py is used throughout the workflow and contains modules that are frequently used.   If there are any questions
about what is implemented here, feel free to contact the authors.
"""


### Functions


def file_opener(FILE, data_dir, idx, delim=','):
    stamp = '%s' % (FILE.split('.')[0])

    with open("%s/%s" % (data_dir, FILE)) as f:
        reader = csv.reader(f, delimiter=delim, skipinitialspace=True)
        data = list(reader)

    start_idx = 0
    index = [int(float(line[0])) for line in data[start_idx:]]
    class_original = [int(float(line[1])) for line in data[start_idx:]]
    class_adversarial = [int(float(line[2])) for line in data[start_idx:]]
    confidence = [line[3] for line in data[start_idx:]]
    layer_values = [line[4:] for line in data[start_idx:]]

    layer_values = [[float(i) for i in line] for line in layer_values]

    if idx == 'no':
        class_names = list(zip(class_original, class_adversarial))
    elif idx == 'yes':
        class_names = list(zip(class_original, class_adversarial, index))

    return stamp, index, class_original, class_adversarial, class_names, confidence, layer_values


def file_opener_predictor(FILE, results_dir):
    stamp = '%s' % (FILE.split('.')[0])

    start_idx = 0
    with open("%s/%s" % (results_dir, FILE)) as f:
        reader = csv.reader(f, delimiter=",", skipinitialspace=True)
        data = list(reader)
    class_original = [int(float(line[0])) for line in data[0:]]
    class_adversarial = [int(float(line[1])) for line in data[0:]]
    index = [int(float(line[2])) for line in data[0:]]
    node_values = [line[3:] for line in data[0:]]

    return stamp, class_original, class_adversarial, index, node_values


def file_subsetter(FILE, data_dir, method, k=5, percent=0.8):
    stamp = '%s' % (FILE.split('.')[0])

    with open("%s/%s" % (data_dir, FILE)) as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        data = list(reader)

    start_idx = 1
    header = data[0]
    index = [int(float(line[0])) for line in data[start_idx:]]
    class_original = [int(float(line[1])) for line in data[start_idx:]]
    class_adversarial = [int(float(line[2])) for line in data[start_idx:]]
    confidence = [line[3] for line in data[start_idx:]]
    layer_values = [line[4:] for line in data[start_idx:]]

    layer_values = [[float(i) for i in line] for line in layer_values]

    subset_size = int(float(percent) * len(data))
    layer_values = np.array(layer_values)

    nbrs = NearestNeighbors(n_neighbors=k).fit(layer_values)
    distances, indices = nbrs.kneighbors(layer_values)

    if method == 'local':
        k_distances = distances[:, k - 1]
        k_density = np.reciprocal(k_distances)
    elif method == 'avg':
        k_distances = [sum(line) for line in distances]
        k_density = np.reciprocal(k_distances)
    elif method == 'avglocal':
        density = np.reciprocal(distances[:, 1:])
        k_density = [sum(line) for line in density]
        k_density = np.array(k_density)

    subset_indices = (-k_density).argsort()[:subset_size]
    data_subset = [line for line in data[start_idx:] if int(float(line[0])) in subset_indices]
    data_subset = np.insert(data_subset, 0, header, 0)

    np.savetxt('%s/%s_avglocalk%s_p%s.csv' % (data_dir, stamp, k, int(10 * percent)), data_subset, delimiter=',',
               fmt='%s')


def filter_show(data):
    for k in [12, 3088]:
        img = np.array(data[k]).reshape((28, 28))

        image_scharr = scharr(img)
        image_sobel = sobel(img)
        image_prewitt = prewitt(img)
        image_gabor_real, image_gabor_im = gabor(img, frequency=0.65)
        image_roberts = roberts(img)
        image_roberts_pos = roberts_pos_diag(img)
        image_roberts_neg = roberts_neg_diag(img)
        image_frangi = frangi(img)
        image_laplace = laplace(img)
        image_hessian = hessian(img)
        image_threshold_local_3 = threshold_local(img, 3)
        image_threshold_local_5 = threshold_local(img, 5)
        image_threshold_local_7 = threshold_local(img, 7)
        image_threshold_niblack = threshold_niblack(img, window_size=5, k=0.1)
        image_threshold_sauvola = threshold_sauvola(img)
        image_threshold_triangle = img > threshold_triangle(img)

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                                 figsize=(8, 8))
        ax = axes.ravel()

        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].set_title('Original image')

        ax[1].imshow(image_threshold_niblack, cmap=plt.cm.gray)
        ax[1].set_title('Niblack')

        ax[2].imshow(image_threshold_sauvola, cmap=plt.cm.gray)
        ax[2].set_title('Sauvola')

        ax[3].imshow(image_threshold_triangle, cmap=plt.cm.gray)
        ax[3].set_title('Triangle')

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()


# fig, ax = try_all_threshold(img)
# plt.tight_layout()
# plt.show()
# plt.close()


def filter(filter_type, layer_values):
    layer_values_processed = []

    def triangle(img):
        image_threshold_triangle = img > threshold_triangle(img)
        image_threshold_triangle = image_threshold_triangle.astype(int)

        return image_threshold_triangle

    def process(data, model):
        for line in layer_values:
            length = len(line)
            size = int(sqrt(length))
            image = np.array(line).reshape(size, size)
            image_processed = model(image).reshape(1, int(length))[0]
            layer_values_processed.append(image_processed)

    def process_2out(data, model, which_out):
        for line in data:
            length = len(line)
            size = int(sqrt(length))
            image = np.array(line).reshape(size, size)
            image_processed_a, image_processed_b = model(image, frequency=0.65)
            if which_out == 'real':
                image_processed = image_processed_a.reshape(1, int(length))[0]
                layer_values_processed.append(image_processed)
            elif which_out == 'imaginary':
                image_processed = image_processed_b.reshape(1, int(length))[0]
                layer_values_processed.append(image_processed)
            elif which_out == 'both':
                image_processed1 = image_processed_a.reshape(1, int(length))[0]
                image_processed2 = image_processed_b.reshape(1, int(length))[0]
                image_processed = np.concatenate((image_processed1, image_processed2))
                layer_values_processed.append(image_processed)

    if filter_type == None:
        layer_values_processed = layer_values
    elif filter_type == 'scharr':
        process(layer_values, scharr)
    elif filter_type == 'gabor_real':
        process_2out(layer_values, gabor, which_out='real')
    elif filter_type == 'gabor_imaginary':
        process_2out(layer_values, gabor, which_out='imaginary')
    elif filter_type == 'gabor_tot':
        process_2out(layer_values, gabor, which_out='both')

    elif filter_type == 'frangi':
        process(layer_values, frangi)
    elif filter_type == 'laplace':
        process(layer_values, laplace)
    elif filter_type == 'hessian':
        process(layer_values, hessian)
    elif filter_type == 'triangle':
        process(layer_values, triangle)

    # image = np.array(layer_values_processed[0], dtype='float')
    # pixels = image.reshape((28, 28))
    # plt.title("Test")
    # plt.imshow(pixels, cmap='gray')
    # plt.show()

    return layer_values_processed


def file_splitter(FILE, data_dir, num_partitions, method='straight'):
    stamp = '%s' % (FILE.split('.')[0])

    with open("%s/%s" % (data_dir, FILE)) as f:
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        data = list(reader)

    data = np.array(data)
    num_points = int(len(data) / num_partitions)

    if method == 'straight':
        split = np.array_split(data, num_partitions)
    elif method == 'random-boot':
        split = []
        for n in range(num_partitions):
            data = ut.shuffle(data)
            split.append(data[:num_points])
    elif method == 'random':
        split = []
        data = ut.shuffle(data)
        for n in range(num_partitions):
            split.append(data[n * num_points:(n + 1) * num_points])

    split = [x.tolist() for x in split]

    for n in range(num_partitions):
        np.savetxt('%s/%s_%ssplit%s.csv' % (data_dir, stamp, method, n + 1), split[n], delimiter=',', fmt='%s')


def histogrammer(data):
    metric_space = sp.spatial.distance.cdist(data, data, metric='euclidean')
    metric_space = np.array(metric_space)
    print(np.amax(metric_space))

    # metric_space = metric_space[:5000,:5000]

    metric_space = ndarray.flatten(metric_space)
    # metric_space = 	metric_space[nonzero(metric_space)]
    # plt.hist(metric_space,bins=10)
    # metric_space = metric_space[np.where( metric_space < 15 )]

    hist, bins = np.histogram(metric_space, bins=10, normed=False)
    width = 1 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    plt.title("Pairwise Distances")
    plt.bar(center, hist, align='center', width=width)
    plt.show()


#	plt.savefig('../images/hist.pdf', bbox_inches='tight', pad_inches=0)


def dendrogrammer(cutoff_data, num_int, num_bin, pcarange):
    # neighbors = kneighbors_graph(data, 1, mode='connectivity', metric='minkowski', p=2)
    # clusters = AgglomerativeClustering(connectivity=None, compute_full_tree='auto', linkage='single').fit(data)

    epsilon = []

    for m in range(1, 11):
        epsilon_split = []
        for k in range(1, 21):

            # filename= open('../data_temp/cutoff_data_temp_trueexamples_in_straightsplit%s_trueexamples_in_straightsplit%s_5_5_PCA%s.txt' %(m,m,k), 'rb')
            # cutoff_data = pickle.load(filename)

            TOTAL_bins = [10]

            for interval in range(1, 11, 1):
                try:
                    for total_bins in TOTAL_bins:

                        data = [line[1] for line in cutoff_data if line[0] == interval]

                        clusters = linkage(data, method='single')
                        metric_space = [line[2] for line in clusters]

                        hist, bins = np.histogram(metric_space, bins=total_bins, normed=False)
                        width = 1 * (bins[1] - bins[0])
                        center = (bins[:-1] + bins[1:]) / 2

                        if 0 not in hist:
                            print('Total Bins %s, Split %s, PCA %s, Interval %s, Cutoff %s' % (
                            total_bins, m, k, interval, bins[-1]))
                            epsilon += float(bins[-1]),
                            epsilon_split += float(bins[-1]),
                        elif 0 in hist:
                            for n, count in enumerate(hist):
                                if count == 0:
                                    print('Total Bins %s, Split %s, PCA %s, Interval %s, Cutoff %s' % (
                                    total_bins, m, k, interval, bins[n]))
                                    epsilon += float(bins[n]),
                                    epsilon_split += float(bins[n]),
                                    break
                except:
                    pass

        print()
        print(len(epsilon))
        print('Mean epsilon for split %s is %s' % (m, mean(epsilon_split)))
        # print 'Final Mean epsilon is %s' %(mean(epsilon))

        plt.title("Single Linkage Cluster")
        plt.bar(center, hist, align='center', width=width)
        plt.savefig('../images/int%s_cluster_%sbin.pdf' % (interval, total_bins), bbox_inches='tight', pad_inches=0)
        plt.close()


'''
	data = data[:50]
	clusters = linkage(data, method='single')
	plt.figure()
	dendrogram(clusters)
	plt.show()
'''


def dendrogrammer2(cutoff_data, num_int, num_bin, pca_component):
    component = pca_component + 1

    for interval in range(1, num_int + 1):

        data = [line[1] for line in cutoff_data if line[0] == interval]

        clusters = linkage(data, method='single')
        metric_space = [line[2] for line in clusters]

        hist, bins = np.histogram(metric_space, bins=num_bin, normed=True)
        width = 1 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        structure_parameter = 0
        if 0 not in hist:
            print('%s Bins, PCA %s, Interval %s, Cutoff %s' % (num_bin, component, interval, bins[-1]))
        elif 0 in hist:
            for n, count in enumerate(hist):
                if count == 0:
                    print('%s Bins, PCA %s, Interval %s, Cutoff %s' % (num_bin, component, interval, bins[n]))
                    break

        plt.title("Single Linkage Cluster PCA%s, Interval %s, %s Bins" % (component, interval, num_bin))
        plt.bar(center, hist, align='center', width=width)
        plt.savefig('../images/pca%s_int%s_cluster_%sbin.pdf' % (component, interval, num_bin), bbox_inches='tight',
                    pad_inches=0)
        plt.close()


def nodler(hist_data, proj_train, proj_test, component, gain, num_int, stamp, stamp_t, idx_test):
    # This is to grab intervals that test points land in
    node_intervals = []
    filter_min = min(proj_train[:, component])
    filter_max = max(proj_train[:, component])

    length = (filter_max - filter_min) / (num_int - (num_int - 1) * gain)
    delta = length * (1 - gain)
    for m in range(num_int):
        lower = filter_min + m * delta
        upper = lower + length
        node_intervals.append([m + 1, lower, upper])

    Hits = []
    WEIGHTS = []
    cutoff_data = []

    for n, entry in enumerate(proj_test[:, component]):
        hits = []
        Weights = []
        Weights_a = []
        if entry < node_intervals[0][1]:  # this pushes points that are less than min to the min
            entry = node_intervals[0][1]
        elif entry >= node_intervals[-1][2]:  # this pushes points that are greater than max to the max
            entry = 0.99 * node_intervals[-1][
                2]  # the 0.99 makes the point slightly lower than edge, which is open interval
        for line in node_intervals:
            if line[1] <= entry < line[2]:
                hits += line[0],

                weight_a = weights(line, entry)  # this is the code i wrote

                weight = jacek_weights(line, entry, line[0], num_int)  # this is the code jacek wrote
                if (weight > 0):
                    hits.append(weight)

                Weights += weight,
                Weights_a += weight_a,

        # cutoff_data.append([line[0],hist_data[n]])
        # hits = hits[:2] # this fixes a weird floating point issue where a point can land in 3 intervalse
        # if len(hits)>2:
        #	print hits
        #	print node_intervals
        #	print Decimal(entry)

        # Hits.append(hits)  #this is the code i wrote

        hitsu = np.unique(hits)  # this is the code jacek wrote
        Hits.append(hitsu)

        WEIGHTS.append(Weights_a)

    WEIGHTS = [[idx_test[n]] + WEIGHTS[n] for n in range(len(idx_test))]

    # return WEIGHTS, cutoff_data, Hits
    return WEIGHTS, Hits


def weights(node_line, entry, stability_param=0.1):
    upper = float(node_line[2])
    lower = float(node_line[1])
    length = upper - lower
    midpoint = lower + float(length) / float(2)
    d = abs(entry - midpoint)
    weight = float(length) / float(d ** 3 + stability_param)

    return weight


def jacek_weights(node_line, entry, cur_index, num_int, gain=0.33, percent=0.2):
    upper = float(node_line[2])
    lower = float(node_line[1])
    length = upper - lower
    midpoint = lower + float(length) / float(2)
    formula = (1 - percent) * (length / 2 - gain * length)
    if (cur_index == 1):
        if (entry - midpoint > formula):
            return cur_index + 1

    elif (cur_index == num_int):
        if (entry - midpoint < - formula):
            return cur_index - 1
    else:
        if (entry - midpoint > formula):
            return cur_index + 1
        if (entry - midpoint < - formula):
            return cur_index - 1
    return -1


def pca(train_data, stamp, num_int, component, idx_test, kpca='no', gain=0.33, scale_features='none', show=False,
        images_dir='../images', results_dir='../results'):
    parse = stamp.split('_')[:-1]
    stamp2 = '_'.join(parse)

    model_name = '../data_temp/pcaprojection_%s.model' % (stamp2)
    # n_components=int(0.5*len(train_data[0]))

    if os.path.exists(model_name) == False:
        if kpca == 'no':
            pca = PCA(whiten=False)
        elif kpca == 'yes':
            pca = KernelPCA(kernel="rbf")
        pca.fit(train_data)
        proj = pca.transform(train_data)
        pickle.dump(pca, open(model_name, 'wb'))
    elif os.path.exists(model_name) == True:
        model_name = open(model_name, 'rb')
        pca = pickle.load(model_name, encoding='latin1')
        proj = pca.transform(train_data)

    # weights, cutoff_data, Hits = nodler(train_data,proj,proj,component,gain,num_int,stamp,stamp,idx_test)
    weights, Hits = nodler(train_data, proj, proj, component, gain, num_int, stamp, stamp, idx_test)

    np.savetxt('../data_temp/temp_%s.txt' % (stamp), proj[:, component], fmt='%s')

    return weights


def pca_test2(train_data, test_data, stamp, stamp_t, num_int, component, idx_test, gain=0.33, scale_features='none',
              show=False, images_dir='../images', results_dir='../results'):
    parse = stamp.split('_')[:-1]
    stamp2 = '_'.join(parse)

    # n_components=int(0.5*len(train_data[0]))
    model_name = open('../data_temp/pcaprojection_%s.model' % (stamp2), 'rb')
    pca = pickle.load(model_name, encoding='latin1')

    pca_name = '../data_temp/projPCA_%s.model' % (stamp2)

    if os.path.exists(pca_name) == False:
        proj = pca.transform(train_data)
        pickle.dump(proj, open(pca_name, 'wb'))
    elif os.path.exists(pca_name) == True:
        pca_name = open(pca_name, 'rb')
        proj = pickle.load(pca_name, encoding='latin1')

    proj_test = pca.transform(test_data)

    # weights, cutoff_data, Hits = nodler(test_data,proj,proj_test,component,gain,num_int,stamp,stamp_t,idx_test)
    weights, Hits = nodler(test_data, proj, proj_test, component, gain, num_int, stamp, stamp_t, idx_test)

    model_name.close()

    return weights, Hits


def lda(train_data, stamp, num_int, component, idx_test, class_original, gain=0.33):
    parse = stamp.split('_')[:-1]
    stamp2 = '_'.join(parse)

    model_name = '../data_temp/ldaprojection_%s.model' % (stamp2)
    # n_components=int(0.5*len(train_data[0]))

    if os.path.exists(model_name) == False:
        lda = LDA()
        lda.fit(train_data, class_original)
        proj = lda.transform(train_data)
        pickle.dump(lda, open(model_name, 'wb'))
    elif os.path.exists(model_name) == True:
        model_name = open(model_name, 'rb')
        lda = pickle.load(model_name, encoding='latin1')
        proj = lda.transform(train_data)

    weights, Hits = nodler(train_data, proj, proj, component, gain, num_int, stamp, stamp, idx_test)

    np.savetxt('../data_temp/temp_%s.txt' % (stamp), proj[:, component], fmt='%s')

    return weights


def lda_test2(train_data, test_data, stamp, stamp_t, num_int, component, idx_test, gain=0.33, scale_features='none',
              show=False, images_dir='../images', results_dir='../results'):
    parse = stamp.split('_')[:-1]
    stamp2 = '_'.join(parse)

    # n_components=int(0.5*len(train_data[0]))
    model_name = open('../data_temp/ldaprojection_%s.model' % (stamp2), 'rb')
    lda = pickle.load(model_name, encoding='latin1')

    lda_name = '../data_temp/projLDA_%s.model' % (stamp2)

    if os.path.exists(lda_name) == False:
        proj = lda.transform(train_data)
        pickle.dump(proj, open(lda_name, 'wb'))
    elif os.path.exists(lda_name) == True:
        lda_name = open(lda_name, 'rb')
        proj = pickle.load(lda_name, encoding='latin1')

    proj_test = lda.transform(test_data)

    # weights, cutoff_data, Hits = nodler(test_data,proj,proj_test,component,gain,num_int,stamp,stamp_t,idx_test)
    weights, Hits = nodler(test_data, proj, proj_test, component, gain, num_int, stamp, stamp_t, idx_test)

    model_name.close()

    return weights, Hits


def contractive_autoencoder(X, lam=1e-4, N_batch=100, N_epoch=100):
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Deconv2D, UpSampling2D, Flatten, Reshape, Dropout
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.regularizers import l1
    from keras.models import load_model
    import keras.backend as K
    import tensorflow as tf
    # tf.python.control_flow_ops = tf

    # K.set_session(tf.Session())

    X = X.reshape(X.shape[0], -1)
    M, N = X.shape

    inputs = Input(shape=(N,))

    # layer = Dense(784, activation='relu')(inputs)
    # layer = Dropout(0.1)(layer)
    encoded = Dense(20, activation='sigmoid', name='encoded')(inputs)
    # layer = Dense(784, activation='relu')(encoded)
    # layer = Dropout(0.1)(layer)

    outputs = Dense(N, activation='linear')(encoded)

    model = Model(inputs=inputs, outputs=outputs)

    def contractive_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

        return mse + contractive

    '''
    inputs = Input(shape=(N,))

    #layer = Dense(784, activation='relu')(inputs)
    #layer = Dropout(0.1)(layer)
    #layer = Dense(392, activation='sigmoid')(inputs)

    layer = Dense(1000, activation='sigmoid')(inputs)
    layer = Dense(500, activation = 'sigmoid')(layer)
    layer = Dense(250, activation = 'sigmoid')(layer)

    encoded = Dense(20, activation='linear', name='encoded')(layer)

    layer = Dense(250, activation = 'sigmoid')(encoded)
    layer = Dense(500, activation = 'sigmoid')(layer)
    layer = Dense(1000, activation = 'sigmoid')(layer)

    #layer = Dense(784, activation='relu')(encoded)
    #layer = Dropout(0.1)(layer)
    #layer = Dense(392, activation='relu')(encoded)
    outputs = Dense(N, activation='linear')(layer)

    model = Model(input=inputs, output=outputs)

    def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)

            W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
            W = K.transpose(W)  # N_hidden x N
            h = model.get_layer('encoded').output
            #dh = h * (1 - h)  # N_batch x N_hidden

            lam = 0
            contractive = 0
            # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
            #contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
            #contractive = lam * K.sum( K.sum(W**2, axis=1) )

            return mse + contractive
    '''
    model.compile(optimizer='adam', loss=contractive_loss)
    model.fit(X, X, batch_size=N_batch, epochs=N_epoch, shuffle=True, verbose=0)

    # K.clear_session()

    return model, Model(inputs=inputs, outputs=encoded)


def auto(train_data, stamp, num_int, component, idx_test, gain=0.33):
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Deconv2D, UpSampling2D, Flatten, Reshape, Dropout
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.regularizers import l1
    from keras.models import load_model
    import keras.backend as K
    import tensorflow as tf
    # tf.python.control_flow_ops = tf

    K.set_session(tf.Session())

    parse = stamp.split('_')[:-1]
    stamp2 = '_'.join(parse)

    model_name = '../data_temp/autoprojection_%s.model' % (stamp2)

    X_train = np.array(train_data)

    if os.path.exists(model_name) == False:
        model, representation = contractive_autoencoder(
            X_train)  # model sends 784 input to 784 output, representation sends 784 input to hidden output
        proj = representation.predict(X_train)
        representation.save(model_name)
    elif os.path.exists(model_name) == True:
        representation = load_model(model_name)
        proj = representation.predict(X_train)

    # weights, cutoff_data, Hits = nodler(train_data,proj,proj,component,gain,num_int,stamp,stamp,idx_test)
    weights, Hits = nodler(train_data, proj, proj, component, gain, num_int, stamp, stamp, idx_test)

    np.savetxt('../data_temp/temp_%s.txt' % (stamp), proj[:, component], fmt='%f')
    K.clear_session()

    return weights, proj


def auto_test2(train_data, test_data, stamp, stamp_t, num_int, component, idx_test, gain=0.33):
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Deconv2D, UpSampling2D, Flatten, Reshape, Dropout
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.regularizers import l1
    from keras.models import load_model
    import keras.backend as K
    import tensorflow as tf

    K.set_session(tf.Session())

    parse = stamp.split('_')[:-1]
    stamp2 = '_'.join(parse)

    model_name = '../data_temp/autoprojection_%s.model' % (stamp2)
    representation = load_model(model_name)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    proj = representation.predict(train_data)
    proj_test = representation.predict(test_data)

    # weights, cutoff_data, Hits = nodler(test_data,proj,proj_test,component,gain,num_int,stamp,stamp_t,idx_test)
    weights, Hits = nodler(test_data, proj, proj_test, component, gain, num_int, stamp, stamp_t, idx_test)

    K.clear_session()

    return weights, Hits, proj, proj_test


def vae(train_data, class_original, test_data, class_original_t, stamp, num_int, component,
        idx_test, gain=0.33, intermediate_dim=512, batch_size=128, latent_dim=20, epochs=100, beta=0):
    from keras.layers import Lambda, Input, Dense
    from keras.models import Model
    from keras.datasets import mnist
    from keras.losses import mse, binary_crossentropy
    from keras.utils import plot_model
    from keras import backend as K
    from keras.models import load_model

    import numpy as np
    import matplotlib.pyplot as plt
    import argparse
    import os

    parse = stamp.split('_')[:-1]
    stamp2 = '_'.join(parse)

    model_name = '../data_temp/vaeprojection_%s.model' % (stamp2)
    projection_data_train = '../data_temp/vaeprojection_%s.traindata' % (stamp2)

    x_train = np.array(train_data)
    x_test = np.array(test_data)

    y_train = np.array(class_original)
    y_test = np.array(class_original_t)

    if os.path.exists(model_name) == False:
        encoder, decoder, vae = variational_autoencoder(x_train, y_train, x_test, y_test, intermediate_dim, batch_size,
                                                        latent_dim, epochs, beta)
        encoder.save(model_name)
        proj = encoder.predict(x_train)
        pickle.dump(proj, open(projection_data_train, 'wb'))
    elif os.path.exists(model_name) == True:
        projection_data_train = open(projection_data_train, 'rb')
        proj = pickle.load(projection_data_train, encoding='latin1')

    # proj[0] is a np array
    # proj[0] has shape 10,000 x 2
    # proj is 3x10000
    # proj[0] is z_mean
    # proj[1] is z_log_var
    # proj[2] is z

    proj = proj[2]

    # weights, cutoff_data, Hits = nodler(x_train,proj,proj,component,gain,num_int,stamp,stamp,idx_test)
    weights, Hits = nodler(x_train, proj, proj, component, gain, num_int, stamp, stamp, idx_test)

    np.savetxt('../data_temp/temp_%s.txt' % (stamp), proj[:, component], fmt='%f')

    return weights, proj


def vae_test2(train_data, test_data, stamp, stamp_t, num_int, component, idx_test, epsilon, gain=0.33):
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Deconv2D, UpSampling2D, Flatten, Reshape, Dropout
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.regularizers import l1
    from keras.models import load_model
    import keras.backend as K
    import tensorflow as tf

    K.set_session(tf.Session())

    parse = stamp.split('_')[:-1]
    stamp2 = '_'.join(parse)
    stamp3 = stamp2 + '_' + stamp_t  # jacek needs to use stamp3 for test block

    model_name = '../data_temp/vaeprojection_%s.model' % (stamp2)
    projection_data_train = '../data_temp/vaeprojection_%s.traindata' % (stamp2)
    projection_data_test = '../data_temp/vaeprojection_%s.testdata_%s' % (
    stamp2, epsilon)  # jacek needs to use stamp3 for test block here

    encoder = load_model(model_name)

    projection_data_train = open(projection_data_train, 'rb')
    proj = pickle.load(projection_data_train, encoding='latin1')

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    if os.path.exists(projection_data_test) == False:
        proj_test = encoder.predict(test_data)
        pickle.dump(proj_test, open(projection_data_test, 'wb'))
    elif os.path.exists(projection_data_test) == True:
        projection_data_test = open(projection_data_test, 'rb')
        proj_test = pickle.load(projection_data_test, encoding='latin1')

    proj = proj[2]
    proj_test = proj_test[2]

    # weights, cutoff_data, Hits = nodler(test_data,proj,proj_test,component,gain,num_int,stamp,stamp_t,idx_test)
    weights, Hits = nodler(test_data, proj, proj_test, component, gain, num_int, stamp, stamp_t, idx_test)

    K.clear_session()

    return weights, Hits, proj, proj_test


def realizer(matrix, weights):
    Matrix = []

    for entry in weights:
        for line in matrix:
            if entry[0] == line[2]:
                temp = []
                j = 1
                for k in line[3:]:
                    if k == 0:
                        temp += 0,
                    elif k == 1:
                        temp += entry[j],
                        j += 1
                Matrix.append(line[:3].tolist() + temp)

    return np.array(Matrix)


def adversary_explorer(y_test, y_predict, index_nodes, data_perturbed, epsilon):
    if float(epsilon) == 1.0:
        test = [y_test != y_predict]
        adversary_line_nums = []
        for n, line in enumerate(test[0]):
            if line == True:
                adversary_line_nums += n,
                print('line %s, node ID %s, class actual %s, class predict %s' % (
                n, index_nodes[n], y_test[n], y_predict[n]))

    # This is to plot what the perturbed data looks like
    for k in adversary_line_nums:
        image = np.array(data_perturbed[k + 1][4:], dtype='float')
        pixels = image.reshape((28, 28))
        plt.legend(loc="upper left")
        plt.title("Test")
        plt.imshow(pixels, cmap='gray')
        plt.show()


def clusterer1(Matrix, Vertex_groups):
    point_clusters = []
    max_vertex_num = Vertex_groups[-1][1]
    unique_vertices = range(1, int(max_vertex_num) + 1)

    for vertex in unique_vertices:
        points_in_vertex = [line[2] for line in Matrix if int(line[int(vertex) + 2]) == 1]
        point_clusters.append([vertex, points_in_vertex])

    return point_clusters


def clusterer2(idx_test_point, vertices, point_clusters, layer_dict, layer_dict_t, k=1):
    test_point = layer_dict_t[idx_test_point]

    idx_train = layer_dict.keys()

    vertex_distances = []
    for vertex in vertices:
        for line in point_clusters:
            if int(vertex) == int(line[0]):
                set_of_points = []
                for idx in line[1]:
                    set_of_points.append(layer_dict[idx])
                dist = sp.spatial.distance.cdist([test_point], set_of_points, metric='euclidean')  # formerly cityblock
                k_nbs = sorted(dist[0])[:k]
                min_dist = mean(k_nbs)

                vertex_distances.append(min_dist)

    closest_vertex = vertices[np.argmin(vertex_distances)]

    return closest_vertex


def clusterer3(Matrix, Matrix_t, layer_dict, layer_dict_t):
    keys = layer_dict.keys()
    layers = layer_dict.values()

    keys_t = layer_dict_t.keys()
    layers_t = layer_dict_t.values()

    new_matrix = []

    dist = sp.spatial.distance.cdist(layers_t, layers, metric='euclidean')

    for n, test_point in enumerate(dist):
        idx_closest_point = keys[np.argmin(dist[n])]
        for line in Matrix:
            if line[2] == idx_closest_point:
                new_matrix.append(np.append(Matrix_t[n], line[3:]))

    print(np.sort(dist[2285])[:3])
    print(np.argsort(dist[2285])[:3])
    print(keys_t[2285])

    closest_train_ids = [keys[n] for n in np.argsort(dist[2285])[:3]]
    layers_subset = [layer_dict[n] for n in closest_train_ids]
    dist_subset = sp.spatial.distance.cdist([layers_t[2285]], layers_subset, metric='euclidean')
    print(dist_subset)
    print(closest_train_ids)

    dist_subset = sp.spatial.distance.cdist(layers_subset, layers_subset, metric='euclidean')
    print(dist_subset)

    new_matrix = np.array(new_matrix)
    return new_matrix


def clusterer_old(Matrix, Matrix_t, layer_dict, layer_dict_t, idx_test_point, vertices, point_clusters, n, k=1):
    # keys = layer_dict.keys()
    # layers = list( layer_dict.values() )

    # keys_t = layer_dict_t.keys()
    # layers_t = list( layer_dict_t.values() )

    test_point = layer_dict_t[idx_test_point]

    idx_subset = []

    for vertex in vertices:
        for line in point_clusters:
            if int(vertex) == int(line[0]):
                for idx in line[1]:
                    idx_subset += idx,

    idx_subset = list(set(idx_subset))

    layers_subset = [layer_dict[idx] for idx in idx_subset]

    dist = sp.spatial.distance.cdist([test_point], layers_subset, metric='euclidean')
    idx_closest_point = idx_subset[np.argmin(dist[0])]

    for line in Matrix:
        if line[2] == idx_closest_point:
            representation_t = np.append(Matrix_t[n], line[3:])

    #	if idx_test_point==0:
    #		print(idx_test_point, idx_closest_point)
    #		print(vertices)
    #		sys.exit()

    return representation_t


def clusterer4(Matrix, Matrix_t, layer_dict, layer_dict_t, index_test_points, vertices, point_clusters, k=20):
    test_points = [layer_dict_t[x] for x in index_test_points]

    idx_subset = []

    for vertex in vertices:
        for line in point_clusters:
            if int(vertex) == int(line[0]):
                for idx in line[1]:
                    idx_subset += idx,

    idx_subset = list(set(idx_subset))

    layers_subset = [layer_dict[idx] for idx in idx_subset]

    dist = sp.spatial.distance.cdist(test_points, layers_subset, metric='euclidean')  # formerly cityblock

    Representation_t = []

    for n, test_point_dists in enumerate(dist):
        idx_test_point = index_test_points[n]

        k_nbs_idx = np.array(test_point_dists).argsort()[:k]
        idx_closest_points = [idx_subset[entry] for entry in k_nbs_idx]

        line_test = np.array([line for line in Matrix_t if int(line[2]) == int(idx_test_point)])

        stability_param = 1e-5
        k_nbs_dist = np.sort(test_point_dists)[:k]
        k_normalize = sum(np.reciprocal(k_nbs_dist + stability_param))
        k_weights = np.reciprocal(k_nbs_dist + stability_param) / k_normalize

        line_train = []  # let's write it this way so that ordering is maintained wrt k_weights
        for index in idx_closest_points:
            for line in Matrix:
                if int(line[2]) == index:
                    line_train.append(line)
                    break

        line_train = np.array(line_train)
        # line_train = np.array([line for line in Matrix if int(line[2])==index in idx_closest_points])
        line_train = line_train[:, 3:]

        weighted_rep = [k_weights[n] * line_train[n] for n in range(len(k_weights))]
        average = np.sum(weighted_rep, axis=0)
        representation_t = np.append(line_test[0], average)
        Representation_t.append(representation_t)

    return Representation_t


def produce_mapper(curr_dir, lib_dir, stamp):
    subprocess.call(['Rscript', '--vanilla', '%s/%s/mnist_mapper_light_%s.r' % (curr_dir, lib_dir, stamp)],
                    stdout=FNULL, stderr=subprocess.STDOUT)


def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])


def matrix_mapper(mapper_file, names, idx):
    with open('../data_temp/%s' % (mapper_file)) as f:
        groups = [line for line in f]
        a = len(groups)
        unique_names = list(set(names))

        vertex_groups = []
        for line in groups:
            interval = line.split(';')[2]
            # size = line.split(';')[3]
            vertex = line.split('\"')[1]
            vertex_groups.append([interval, vertex])
        del (vertex_groups[0])

        if idx == 'no':
            matrix = [[i[0], i[1]] for i in unique_names]
        elif idx == 'yes':
            matrix = [[i[0], i[1], i[2]] for i in unique_names]

        n = -1
        for line in unique_names:
            n += 1
            for i in range(a - 1):
                group_names = groups[i + 1].split(':')[1].split(';')[0].split("\"")[0].split(',')
                if idx == 'no':
                    matrix[n] += group_names.count(' %s %s' % (line[0], line[1])),
                elif idx == 'yes':
                    matrix[n] += group_names.count(' %s %s %s' % (line[0], line[1], line[2])),

    return np.vstack(matrix), np.vstack(vertex_groups)


def matrix_squarer(data):
    matrix = [line[1:] for line in data]
    matrix = np.asarray(matrix)
    square = np.dot(matrix, matrix.T)
    return square


def norm(data1, data2):
    matrix1 = np.asarray(data1)
    matrix2 = np.asarray(data2)
    return LA.norm(matrix1 - matrix2)


'''
def pca_test(train_data,test_data,stamp,num_int,component,gain=0.33,scale_features='none',show=False,images_dir='../images',results_dir='../results'):
	stamp2=stamp[:-5]
	n_components=int(0.5*len(train_data[0]))
	model_name = open('../data_temp/picklemePCA_%s.model' %(stamp2), 'rb')
	pca = pickle.load(model_name)

	if scale_features == 'scale':
		train_data = preprocessing.scale(train_data)
		test_data = preprocessing.scale(test_data)
	elif scale_features == 'maxabs':
		train_data = preprocessing.maxabs_scale(train_data)
		test_data = preprocessing.maxabs_scale(test_data)
	elif scale_features == 'minmax':
		train_data = preprocessing.minmax_scale(train_data)
		test_data = preprocessing.minmax_scale(test_data)
	elif scale_features == 'normalize':
		train_data = preprocessing.normalize(train_data)
		test_data = preprocessing.normalize(test_data)
	elif scale_features == 'robust':
		train_data = preprocessing.robust_scale(train_data)
		test_data = preprocessing.robust_scale(test_data)

	pca_name = '../data_temp/projPCA_%s.model' %(stamp2)

	if os.path.exists(pca_name) == False:
		proj = pca.transform(train_data)
		pickle.dump(proj, open(pca_name, 'wb'))
	elif os.path.exists(pca_name) == True:
		pca_name=open(pca_name)
		proj = pickle.load(pca_name)

	proj_test = pca.transform(test_data)


	node_intervals = nodler(proj,proj_test,component,gain,num_int)


	Hits=[]
	for entry in proj_test[:,component]:
		hits=[]
		if entry < node_intervals[0][1]: # this pushes points that are less than min to the min
			entry = node_intervals[0][1]
		elif entry >= node_intervals[-1][2]: # this pushes points that are greater than max to the max
			entry = node_intervals[-1][2]
		for line in node_intervals:
			if line[1]<=Decimal(entry)<line[2]:
				hits+=line[0],
		hits = hits[:2] # this fixes a weird floating point issue where a point can land in 3 intervals
		if len(hits)>2:
			print hits
			print node_intervals
			print Decimal(entry)
		Hits.append(hits)
	filename='../data_temp/test_nodes_temp_%s.txt' %(stamp)
	pickle.dump(Hits, open(filename, 'wb'))
'''
