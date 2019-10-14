#!python

from numpy import *
import matplotlib.pyplot as plt
import os, string, sys, shutil
import numpy as np
import math
import csv
import numbers
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib.patches import FancyArrowPatch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import xml.etree.ElementTree as ET
import yaml
import json
import codecs
import glob
from ast import literal_eval
import scipy as sp
import subprocess
from numpy import linalg as LA
from itertools import chain
from tools import *
import pickle
from scipy.spatial import distance
from skimage.filters import scharr, gabor

FNULL = open(os.devnull, 'w')
curr_dir = os.getcwd()


def pymapper(FILE_TRAIN, FILE_TEST, data_dir, data_dir_test, dimensions, epsilon=0, projection='pca', filter_type=None,
             real_matrix=None, train='yes', map_method='cluster4', component=1, num_int=10, num_bin=10, idx='yes',
             scale_features='none', results_dir='../results', lib_dir='../lib'):
    ###################################################################
    # Section 1A: Grabs Data and setup for runs
    ###################################################################

    """
    pymapper() is more or less the central function that does all the heavy lifting in the entire workflow.  We
    outsource the actual mapper production to an R package - this will change in future versions of the code as we
    will implement this procedure ourselves.

    :param FILE_TRAIN: Raw data for training.  For instance, for MNIST, this consists of the pixel values for each
    data point (i.e. image).
    :param FILE_TEST: Raw data for testing.
    :param data_dir: Where we will grab above FILE_TRAIN
    :param data_dir_test: Where we will grab above FILE_TEST
    :param dimensions: Will we build 1-d or 2-d mappers?  Old versions of this code could produce both.  We've taken
    out 2-d capabilities, but will build back in future versions.
    :param epsilon: Noise parameter.  Only useful when running VAE adversary().  It's a bad parameter that can be
    worked out of this code by re-working VAE implementation.  Feel free to contact authors.
    :param projection: What kind of projection will we do?  See tools.py for the various methods: {pca, lda, auto,
    vae}.  Additionally, these models can be tweaked to return other similar models.
    :param filter_type: Will we run the initial raw data (FILE_TRAIN, FILE_TEST) through a preprocessing filter?
    Again, see tools.py
    :param real_matrix: Will we allow FILE_TRAIN to have real value?  Or will it just take on the usual binary vals?
    real_matrix=False means usual binary vals.
    :param train: Toggles whether we are currently training or testing: {yes, no}. train='no' means we are testing.
    I know, bad implementation.
    :param map_method:  How do we map test data points through the mapper committee?  We've implemented a number of
    different ways - feel free to contact authors about this.  Current best method is clusterer4(), which can be
    found in tools.py
    :param component: Which axis of the projection we'll use to create mapper
    :param num_int: # intervals to use in the open cover.
    :param num_bin: # bins to use in the hist when clustering.
    :param idx: Whether or not unique identifiers should be placed on data points.  99% of the time, keep to 'yes'.
    :param scale_features: Another preprocessing of data option.  See tools.py
    :param results_dir: Where will we save produced matrix mapper objects?
    :param lib_dir: Where tools.py will be located
    :return: Trained or tested matrix mapper objects, in csv format.
    """


    # Might want to toggle below two lines if having issues with parallel training.  Contact one of the authors if
    # you'd like to learn more.
    # if component in [2, 3, 4]:
    #    time.sleep(600)

    # Grab important metadata from file
    # If scaling needs done, do so directly using something like: pca(layer_values,stamp,scale_features='minmax')


stamp, index, class_original, class_adversarial, class_names, confidence, layer_values = file_opener(FILE_TRAIN,
                                                                                                     data_dir, idx)
stamp_t, index_t, class_original_t, class_adversarial_t, class_names_t, confidence_t, layer_values_t = file_opener(
    FILE_TEST, data_dir_test, idx, ',')

layer_values = filter(filter_type, layer_values)
layer_values_t = filter(filter_type, layer_values_t)

layer_dict = {}
for i in range(len(index)):
    layer_dict[index[i]] = layer_values[i]

layer_dict_t = {}
for i in range(len(index_t)):
    layer_dict_t[index_t[i]] = layer_values_t[i]

# for k in range(10):
# image = np.array(layer_values[k], dtype='float')
# pixels = image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()

# Print dimensionality & number of points i  data set.  Also save info about this data.
# print 'Dimensionality: %s' %(len(layer_values[0]))
# print 'Entries: %s' %(len(layer_values))

'''
metric_space = sp.spatial.distance.cdist(layer_values, layer_values, metric='euclidean')
metric_space = np.array(metric_space)
metric_space = ndarray.flatten(metric_space)
metric_space = 	metric_space[nonzero(metric_space)]
print(np.amax(metric_space))
print(mean(metric_space))
print(np.amin(metric_space))
print(mean(np.reciprocal(metric_space)))
'''
Component = component - 1  # added this since component is changed below to string format.  This keeps it as an int.
component = 'PROJ%s' % (component)
stamp = '%s_%s_%s_%s' % (stamp, num_int, num_bin, component)

###################################################################
# Section 2: Compute M(x) for training.
###################################################################
if train == 'yes':

    # Simply prepare the mapper script to be run later
    if dimensions == 1:
        subprocess.call(
            "sed -e \"s/stamp/%s/g\" -e \"s/NINT/%d/g\" -e \"s/NBIN/%d/g\" -e \"s/PCACOMP/%s/g\" \"%s/%s/mnist_mapper_light_template1D.r\" > \"%s/%s/mnist_mapper_light_%s.r\""
            % (stamp, int(num_int), int(num_bin), component, curr_dir, lib_dir, curr_dir, lib_dir, stamp),
            shell=True)

    np.savetxt('../data_temp/%s_names.txt' % (stamp), class_names, fmt='%s')
    np.savetxt('../data_temp/%s_layers.txt' % (stamp), layer_values, fmt='%s')

    if projection == 'pca':
        weights = pca(layer_values, stamp, num_int, Component, index)
    elif projection == 'cae':
        weights, proj = auto(layer_values, stamp, num_int, Component, index)
    # weights = pca(proj,stamp,num_int,Component,index)
    elif projection == 'vae':
        weights, proj = vae(layer_values, class_original, layer_values_t, class_original_t, stamp, num_int,
                            Component, index_t)
    # weights = pca(proj,stamp,num_int,Component,index)
    elif projection == 'lda':
        weights = lda(layer_values, stamp, num_int, Component, index, class_original)

    # dendrogrammer2(cutoff_data, num_int, num_bin,Component)
    produce_mapper(curr_dir, lib_dir, stamp)
    Matrix, Vertex_groups = matrix_mapper('%s_nodes.txt' % (stamp), class_names, idx)
    if real_matrix == 'yes':
        Matrix = realizer(Matrix, weights)
    if len(Matrix[0] < 100):
        np.savetxt('%s/matrix_%s.csv' % (results_dir, stamp), Matrix, delimiter=', ', fmt='%f')
    elif len(Matrix[0] >= 100):
        return -1

###################################################################
# Section 3: Compute M(x) for testing.  Just a 1D implementation for now.
###################################################################

elif train == 'no':

    if map_method == 'pca':

        weights_t, intervals = pca_test2(layer_values, layer_values_t, stamp, stamp_t, num_int, Component, index_t,
                                         scale_features=scale_features)
        Matrix, Vertex_groups = matrix_mapper('%s_nodes.txt' % (stamp), class_names, idx)

        # test_file = '../data_temp/test_nodes_temp_%s_%s.txt' %(stamp_t, stamp)

        Matrix_t = np.vstack(class_names_t)

        # filename=open('../data_temp/test_nodes_temp_%s_%s.txt' %(stamp_t, stamp))
        # data = pickle.load(filename)

        point_clusters = clusterer1(Matrix, Vertex_groups)
        # point_clusters[i][0] is the vertex & data type = int
        # point_clusters[i][1] is the set of points & data type = list

        max_vertex_num = Vertex_groups[-1][1]

        vertex_t = []
        n = -1
        for line in intervals:
            n += 1
            idx_test_point = index_t[n]
            clusters = []
            for entry in line:
                vertices = [int(line[1]) for line in Vertex_groups if int(entry) == int(line[0])]
                if len(vertices) == 1:
                    clusters += vertices[0],
                elif len(vertices) > 1:
                    clusters += clusterer2(idx_test_point, vertices, point_clusters, layer_dict, layer_dict_t),
            vertex_t.append(clusters)

        # for line in vertex_t:
        #	if len(line)>2:
        #		print(line, '%s' %(test_file))

        vertex_matrix_t = []
        for line in vertex_t:
            vector = [0] * int(max_vertex_num)
            for entry in line:
                vector[entry - 1] = 1
            vertex_matrix_t.append(vector)
        vertex_matrix_t = np.array(vertex_matrix_t)

        Matrix_t = np.concatenate((Matrix_t, vertex_matrix_t), axis=1)

        if real_matrix == 'yes':
            Matrix_t = realizer(Matrix_t, weights_t)

        np.savetxt('%s/test_matrix_%s_%s.csv' % (results_dir, stamp_t, stamp), Matrix_t, delimiter=', ', fmt='%f')

    elif map_method == 'cluster':

        Matrix, Vertex_groups = matrix_mapper('%s_nodes.txt' % (stamp), class_names, idx)
        Matrix_t = np.vstack(class_names_t)

        Matrix_t = clusterer3(Matrix, Matrix_t, layer_dict, layer_dict_t)

        np.savetxt('%s/test_matrix_%s_%s.csv' % (results_dir, stamp_t, stamp), Matrix_t, delimiter=', ', fmt='%f')


    elif map_method == 'cluster4':
        if projection == 'pca':  # pca projection with k = 1 is the same as map_method = pca
            weights_t, intervals = pca_test2(layer_values, layer_values_t, stamp, stamp_t, num_int, Component,
                                             index_t)
        elif projection == 'cae':
            weights_t, intervals, proj, proj_test = auto_test2(layer_values, layer_values_t, stamp, stamp_t,
                                                               num_int, Component, index_t)
        # weights_t, intervals = pca_test2(proj,proj_test,stamp, stamp_t, num_int,Component,index_t)
        elif projection == 'vae':
            weights_t, intervals, proj, proj_test = vae_test2(layer_values, layer_values_t, stamp, stamp_t, num_int,
                                                              Component, index_t, epsilon)
        # weights_t, intervals = pca_test2(proj,proj_test,stamp, stamp_t, num_int,Component,index_t)
        elif projection == 'lda':
            weights_t, intervals = lda_test2(layer_values, layer_values_t, stamp, stamp_t, num_int, Component,
                                             index_t)

        Matrix, Vertex_groups = matrix_mapper('%s_nodes.txt' % (stamp), class_names, idx)
        test_file = '../data_temp/test_nodes_temp_%s_%s.txt' % (stamp_t, stamp)
        Matrix_t = np.vstack(class_names_t)

        # filename=open('../data_temp/test_nodes_temp_%s_%s.txt' %(stamp_t, stamp))
        # data = pickle.load(filename)

        point_clusters = clusterer1(Matrix, Vertex_groups)
        # point_clusters[i][0] is the vertex & data type = int
        # point_clusters[i][1] is the set of points & data type = list

        with open('%s/matrix_%s.csv' % (results_dir, stamp)) as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            Matrix = list(reader)

        Matrix = np.asarray(Matrix, dtype=np.float64)

        Vertices = []

        Matrix_new = []

        for n, line in enumerate(intervals):
            vertices = []
            for entry in line:
                for blank in Vertex_groups:
                    if int(entry) == int(blank[0]):
                        vertices += int(blank[1]),
            Vertices.append(vertices)
        Vertices_unique = [list(x) for x in set(tuple(x) for x in Vertices)]

        for unique_vertices in Vertices_unique:
            index_test_points = []
            for n, vertices in enumerate(Vertices):
                if vertices == unique_vertices:
                    index_test_points += index_t[n],

            representation_t = clusterer4(Matrix, Matrix_t, layer_dict, layer_dict_t, index_test_points,
                                          unique_vertices, point_clusters)
            for line in representation_t:
                Matrix_new.append(line)

        Matrix_ordered = []

        for entry in Matrix_t:
            for line in Matrix_new:
                if entry[2] == line[2]:
                    Matrix_ordered.append(line)

        np.savetxt('%s/test_matrix_%s_%s.csv' % (results_dir, stamp_t, stamp), Matrix_ordered, delimiter=', ',
                   fmt='%f')
