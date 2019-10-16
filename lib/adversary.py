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
from multiprocessing import Process
from multiprocessing import Pool

FNULL = open(os.devnull, 'w')
curr_dir = os.getcwd()

"""
This script is used to create adversarial examples, then test how your trained classifier performs.
One output is a report that includes various statistics, including the recall & precision, at each noise level.
The other output is a recorded log of the l2-norms - i.e. the distance at which an instance is finally misclassified.
"""


def test_block(argsl):
    """
    This function simply allows us to run multiple pymapper instances in parallel (i.e. multiple mapper objects will
    be computed concurrently).
    :param argsl: Arguments set by adversary() below.
    :return: A run over the pymapper() function.
    """

    k = argsl[0]
    epsilon = argsl[1]
    classification_model = argsl[2]
    noise_model = argsl[3]
    map_method = argsl[4]
    kpca = argsl[5]
    real_matrix = argsl[6]
    num_int = argsl[7]
    num_bin = argsl[8]
    filter_type = argsl[9]
    file_train = argsl[10]

    pymapper(file_train, 'data_perturbed_%s_%s_%s.csv' % (int(epsilon), classification_model, noise_model),
             '../data_train_fashion_unnormalized', '../data_temp', train='no',
             epsilon=int(epsilon), map_method=map_method, kpca=kpca,
             real_matrix=real_matrix, dimensions=1, component=k, num_int=num_int,
             num_bin=num_bin, idx='yes', filter_type=filter_type)


def adversary(data_dir_nodes, test_data_nodes, num_int, num_bin, PCArange, file_train,
              real_matrix=None, filter_type=None, noise_model='gauss_blur', idx='yes', map_method='pca',
              kpca='no', images_dir='../images', classification_model='rf'):
    """
    Adversary is the function that does all the heaving lifting here.  It is the last step in the DAG of the whole process.  Please see README.txt for the overall workflow.

    Please be aware that there are numerous places where we hard-code.

    YOU MUST TRAIN UP YOUR NN OR END CLASSIFIER BEFORE RUNNING ADVERSARY.  A BEGINNING STEP BELOW IS TO LOAD IN YOUR
    TRAINED MODEL.

    :param data_dir_nodes: Directory where the merged matrix mapper information is stored.  This is a merged csv file, output by the joiner() function in lib/
    :param test_data_nodes: The name of the merged matrix mapper csv.
    :param num_int: # intervals to use in the open cover.
    :param num_bin: # bins to use in the hist when clustering.
    :param PCArange: # components to use in the projection.  Default is 20 components.  'PCArange' still applies for other types of projections - just bad wording.
    :param file_train: name of the training data.
    :param real_matrix: A toggle feature which allows floating value for trained matrix mapper.  'yes' means real values.
    :param filter_type: If filter preprocessing should be done.  The filters that are implemented are in tools()
    :param noise_model: What kind of noise we'll add: {'rand_int','gauss_blur','s&p','gaussian','shift'}
    :param idx: Whether or not unique identifiers should be placed on data points.  99% of the time, keep to 'yes'.
    :param map_method: The projection model we use.
    :param kpca: Whether or not to use kpca.  Written badly.  Probably disregard this.
    :param images_dir:  Where to save images.
    :param classification_model: The type of model we will use as the end classifier.
    :return:
        (1) report that includes various statistics, including the recall & precision, at each noise level.
            this information currently prints to screen.
        (2) recorded log of the l2-norms - i.e. the distance at which an instance is finally misclassified.
            file name is something like: l2norms_mlp2_gauss_blur.csv
    """

    ###################################################################################################################
    ###################################################################################################################

    data_dir_test = '../data_test_fashion_unnormalized'  # this is the location of the raw test data (i.e. the data with all the pixel values)
    test_data = 'trueexamples_in.csv'  # this is the name of the raw test data (i.e. the data with all the pixel values)

    # data_dir_nodes = '../results/round38_5_10_pca_33gain'		# this is the location of the node test data (i.e. the binary mapper representation of test points)
    # test_data_nodes = 'test_merged_mappers_all_20.csv'	# this is the name of the node test data

    model_name = open('../data_temp/%s.model' % (classification_model),
                      'rb')  # this is the name of the classification model being used
    model = pickle.load(model_name)  # this loads the classification model

    stamp_data, index_data, class_original_data, class_adversarial_data, class_names_data, confidence_data, layer_values_data = file_opener(
        test_data, data_dir_test, idx, ',')
    stamp_nodes, class_original_nodes, class_adversarial_nodes, index_nodes, node_values = file_opener_predictor(
        test_data_nodes, data_dir_nodes)

    class_original = class_original_data

    # Find initial correctly classified instances with model
    # Data comes from nodes test data

    X_test = np.array(node_values).astype(float)
    y_test = np.array(class_original_nodes).astype(float)

    y_predict = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    report = classification_report(y_test, y_predict)

    y_predict = y_predict.flatten()

    print(accuracy, report)

    INITIAL_ACC = accuracy
    TOTAL_MISCLASS = 0

    # Extract out raw data of correctly classified instances
    # Data comes from raw test data
    layer_values_data = np.array(layer_values_data).astype(float)
    print(len(y_test), len(y_predict))

    Classified_Data = layer_values_data[y_test == y_predict]

    Class_Original_Data = np.array(class_original_data).astype(float)
    Index_Data = np.array(index_data).astype(float)
    Y_Classified = Class_Original_Data[y_test == y_predict]
    Index_Classified = Index_Data[y_test == y_predict]

    outfnorms = open('../results/l2norms_%s_%s.csv' % (classification_model, noise_model), 'w')

    '''
    # This is to plot what the image data looks like
    image = np.array(Classified_Data[0], dtype='float')
    pixels = image.reshape((28, 28))
    plt.title("Gauss Blur")
    plt.imshow(pixels, cmap='gray')
    plt.savefig('%s/%s_0.pdf' %(images_dir,noise_model), bbox_inches='tight', pad_inches=0)
    '''

    # Add noise to raw data of correctly classified instances
    # Data comes from raw test data
    # Do some stupid data preparing to be saved to text
    # Ultimately this is necessary so file read-in/out is consistent
    # Activate = yes means reproduce perturbing procedure
    # Activate = no means use last perturbed setup
    # 	--> (mainly to produce new images without having to go through a whole new perturbing procedure each time)

    if noise_model == 'rand_int':
        epsilons = np.linspace(0, 42, 7)[1:]
    elif noise_model == 'gauss_blur':
        epsilons = np.linspace(0, 6, 7)[1:]
    elif noise_model == 's&p':
        epsilons = np.linspace(0, 60, 7)[1:]
    elif noise_model == 'gaussian':
        epsilons = np.linspace(0, 6, 7)[1:]
    elif noise_model == 'shift':
        epsilons = np.linspace(0, 50, 11)[1:]

    L2 = []
    exclude_data = []
    l2_format = []

    for epsilon in epsilons:
        print(epsilon)
        # if int(epsilon)==140:
        #	break

        l2 = []

        data_perturbed = []
        test1 = Classified_Data[0]

        if noise_model == 'rand_stupid':
            noise = np.random.rand(len(Classified_Data), len(Classified_Data[0]))
            noise = 0.01 * epsilon * noise  # 0.01 here is a silly workaround for a convention in filenames
            data_perturbed = noise + Classified_Data
        elif noise_model == 'rand_int':
            for image in Classified_Data:
                length = len(image)
                size = int(sqrt(length))
                min_ = min(image)
                max_ = max(image)
                w = max_ - min_
                image = np.array(image).reshape(size, size)
                noise = np.random.uniform(-w, w, size=image.shape)
                perturbed = image + 0.01 * epsilon * noise
                perturbed = perturbed.reshape(1, int(length))[0]
                perturbed = np.clip(perturbed, min_, max_)
                data_perturbed.append(perturbed)
        elif noise_model == 'gauss_blur':
            for image in Classified_Data:
                length = len(image)
                size = int(sqrt(length))
                min_ = min(image)
                max_ = max(image)
                image = np.array(image).reshape(size, size)
                sigma = 0.01 * epsilon * size
                blurred = gaussian_filter(image, sigma)
                blurred = blurred.reshape(1, int(length))[0]
                blurred = np.clip(blurred, min_, max_)
                data_perturbed.append(blurred)
        elif noise_model == 's&p':
            for image in Classified_Data:
                length = len(image)
                size = int(sqrt(length))
                min_ = min(image)
                max_ = max(image)
                w = max_ - min_
                image = np.array(image).reshape(size, size)
                p = 0.001 * epsilon  # amount of salt & pepper
                q = 0.5  # how much salt vs pepper
                out = image.copy()
                flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
                salted = np.random.choice([True, False], size=image.shape, p=[q, 1 - q])
                peppered = ~salted
                out[flipped & salted] = max_
                out[flipped & peppered] = min_
                out = out.reshape(1, int(length))[0]
                data_perturbed.append(out)
        elif noise_model == 'gaussian':
            for image in Classified_Data:
                length = len(image)
                size = int(sqrt(length))
                min_ = min(image)
                max_ = max(image)
                w = max_ - min_
                image = np.array(image).reshape(size, size)
                noise = np.random.normal(0, 0.1 * (epsilon) ** 0.5, image.shape)
                out = image + noise
                out = out.reshape(1, int(length))[0]
                out = np.clip(out, min_, max_)
                data_perturbed.append(out)
        elif noise_model == 'shift':
            for image in Classified_Data:
                length = len(image)
                size = int(sqrt(length))
                min_ = min(image)
                max_ = max(image)
                out = [float(1 + 3 * 0.01 * epsilon) * float(pixel) - float(epsilon) for pixel in image]
                data_perturbed.append(out)

        # This block is to plot what the perturbed data looks like
        '''
        image = np.array(data_perturbed[0], dtype='float')
        pixels = image.reshape((28, 28))
        plt.title("Gauss Blur")
        plt.imshow(pixels, cmap='gray')
        plt.savefig('%s/%s_%s.pdf' %(images_dir,noise_model,int(epsilon)), bbox_inches='tight', pad_inches=0)
        plt.show()
        test2 = data_perturbed[0]
        print(sp.spatial.distance.euclidean(test1,test2))
        '''

        padding = np.zeros((len(data_perturbed), 1))
        data_perturbed = np.column_stack((padding, data_perturbed))
        data_perturbed = np.column_stack((Y_Classified, data_perturbed))
        data_perturbed = np.column_stack((Y_Classified, data_perturbed))
        data_perturbed = np.column_stack((Index_Classified, data_perturbed))
        header = np.ones((1, len(data_perturbed[0])))
        header = - header

        indices = []
        for n, entry in enumerate(Index_Classified):
            if entry in exclude_data:
                indices += n,

        data_perturbed = np.delete(data_perturbed, indices, 0)

        np.savetxt('../data_temp/data_perturbed_%s_%s_%s.csv' % (int(epsilon), classification_model, noise_model),
                   data_perturbed, delimiter=',', fmt='%s')

        # Run perturbed data through pymapper to construct new test matrix
        # And then merge all the results

        ########################################################################################################################
        # CHANGE FOLLOWING FEW LINES TO:
        # 1) MAKE PATHS APPROPRIATE.  SPECIFICALLY PYMAPPER PATHS MIGHT NEED TO BE CHANGED.
        # 2) ADD IN YOUR JOINER PROCEDURE THAT OPERATES OVER SPLITS
        ########################################################################################################################
        temp_stamp = '%s' % (file_train.split('.')[0])
        file_name = '../results/test_merged_mappers_all_%s_%s_%s.csv' % (
            int(epsilon), classification_model, noise_model)

        ARGS = [epsilon, classification_model, noise_model, map_method, kpca, real_matrix, num_int, num_bin,
                filter_type, file_train]
        argsl = [[k] + ARGS for k in PCArange]

        PROCS = 4
        pool = Pool(PROCS)
        if os.path.exists(file_name) == False:
            pool.map(test_block, argsl, chunksize=1)
            joiner_adversary('../results', int(epsilon), classification_model, noise_model, num_int, num_bin, PCArange,
                             temp_stamp)
        pool.close()

        test_data_nodes = 'test_merged_mappers_all_%s_%s_%s.csv' % (int(epsilon), classification_model, noise_model)
        data_dir_nodes = '../results'
        stamp_nodes, class_original_nodes, class_adversarial_nodes, index_nodes, node_values = file_opener_predictor(
            test_data_nodes, data_dir_nodes)

        X_test = np.array(node_values).astype(float)
        y_test = np.array(class_original_nodes).astype(float)

        y_predict = model.predict(X_test)
        y_predict = y_predict.flatten()

        accuracy = model.score(X_test, y_test)
        report = classification_report(y_test, y_predict)

        print(report)

        if accuracy <= 0.1:  # at this point, we're randomly guessing, so we shouldn't artificially inflate our robustness measure
            print('accuracy is now %s which is below or equal to random guessing' % (accuracy))
            break

        # Finally, let's see how the adversarial/non-adversarial images actually look
        stamp, index, class_true, class_adversarial, class_names, confidence, layer_values = file_opener(
            'data_perturbed_%s_%s_%s.csv' % (int(epsilon), classification_model, noise_model), '../data_temp', idx, ',')

        layer_values = np.array(layer_values).astype(float)
        class_true = np.array(class_true).astype(float)
        index = np.array(index).astype(float)

        data_classified = layer_values[y_test == y_predict]
        data_misclassified = layer_values[y_test != y_predict]

        y_classified = class_true[y_test == y_predict]
        y_misclassified = class_true[y_test != y_predict]

        index_classified = index[y_test == y_predict]
        index_misclassified = index[y_test != y_predict]

        exclude_data += ndarray.tolist(index_misclassified)

        data_original_classified = []
        data_original_misclassified = []
        for n, index in enumerate(Index_Classified):
            if index in index_classified:
                data_original_classified.append(Classified_Data[n])
            elif index in index_misclassified:
                data_original_misclassified.append(Classified_Data[n])

        TOTAL_MISCLASS += len(index_misclassified)

        for n in range(len(data_original_misclassified)):
            dist = sp.spatial.distance.euclidean(data_original_misclassified[n], data_misclassified[n])
            L2.append(dist)
            l2.append(dist)

        np.savetxt(outfnorms, np.reshape([0.01 * epsilon] + l2, (1, -1)), delimiter=',')

        try:
            print('Total Instances = %s; Num instances misclassified = %s; Minimal noise = %s; Accuracy = %s' % (
                len(data_perturbed), len(index_misclassified), float(0.01 * epsilon), accuracy))
            print('Average L2 Norm = %s; Min L2 Norm = %s' % (mean(l2), min(l2)))
            ratio = TOTAL_MISCLASS / INITIAL_ACC
            ratio = round(ratio, 4)
            print('Robustness Ratio = %s' % (ratio))
        except:
            print('Your algorithm is hella good')
    try:
        print('Total Average L2 Norm = %s; Total Min L2 Norm =  %s' % (mean(L2), min(L2)))
    except:
        print('Your algorithm is hella good')
