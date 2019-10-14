#!python

import pandas as pd
from os import listdir
from os.path import isfile, join


def joiner(results_dir, num_int, num_bin, PCArange, stamp):
    """
    This function merges matrix mapper objects together (i.e. a mapper object will be produced for each dimension we
    project to, and joiner() merges each of these.  By default, we project to 20 dimensional space in the initial
    projection, so we have 20 mapper objects to merge.  'PCArange' is a bad name, as this variable merges any
    projection type - PCArange is just is a list of integers.

    :param results_dir: This is the directory each matrix mapper object will be saved (these are csv's)
    :param num_int: # intervals to use in the open cover.
    :param num_bin: # bins to use in the hist when clustering.
    :param PCArange: # components to use in the projection.  Default is 20 components.  'PCArange' still applies for
    other types of projections - just bad wording.
    :param stamp: a unique identifier that is useful for tracking, and writing out files.
    :return:
        (1) A csv that contains merged information over all trained matrix mappers
        (2) A csv that contains merged information over all test matrix mappers

    """

    ## Join trained matrix mappers
    files = ['matrix_%s_%s_%s_PROJ%s.csv' % (stamp, num_int, num_bin, k) for k in PCArange]

    n = 0
    FILES = []
    for entry in files:
        FILES += entry,
        n += 1

        df = {}
        for file in FILES:
            df[file] = pd.read_csv('%s/%s' % (results_dir, file), sep=',', header=None, index_col=[0, 1, 2])

        result = pd.concat([df[x] for x in FILES], axis=1)
        result.to_csv('%s/merged_mappers_all_%s.csv' % (results_dir, n), header=False)

    ## Join test matrix mappers
    files = ['test_matrix_trueexamples_in_%s_%s_%s_PROJ%s.csv' % (stamp, num_int, num_bin, k) for k in PCArange]

    n = 0
    FILES = []
    for entry in files:
        FILES += entry,
        n += 1

        df = {}
        for file in FILES:
            df[file] = pd.read_csv('%s/%s' % (results_dir, file), sep=',', header=None, index_col=[0, 1, 2])

        result = pd.concat([df[x] for x in FILES], axis=1)
        result.to_csv('%s/test_merged_mappers_all_%s.csv' % (results_dir, n), header=False)


def joiner_adversary(results_dir, epsilon, model, noise_model, num_int, num_bin, PCArange, stamp):
    """
    Function is same as above, though with the additional noise parameter.  We will have a different set of matrix
    objects @ each noise level.

    :param results_dir:
    :param epsilon:
    :param model:
    :param noise_model:
    :param num_int:
    :param num_bin:
    :param PCArange:
    :param stamp:
    :return:
    """

    ## Join test matrix mappers
    files = ['test_matrix_data_perturbed_%s_%s_%s_%s_%s_%s_PROJ%s.csv' % (
        epsilon, model, noise_model, stamp, num_int, num_bin, k) for k in PCArange]

    df = {}
    for file in files:
        df[file] = pd.read_csv('%s/%s' % (results_dir, file), sep=',', header=None, index_col=[0, 1, 2])

    result = pd.concat([df[x] for x in files], axis=1)
    result.to_csv('%s/test_merged_mappers_all_%s_%s_%s.csv' % (results_dir, epsilon, model, noise_model), header=False)


'''
def joiner_adversary2(results_dir,epsilon,model,noise_model,num_int,num_bin,PCArange):

	## Join test matrix mappers in the iterative fashion like above
	files = ['test_matrix_data_perturbed_%s_%s_%s_trueexamples_in_%s_%s_PROJ%s.csv' %(epsilon,model,noise_model,num_int,num_bin,k) for k in PCArange]

	n=0
	FILES=[]
	for entry in files:
		FILES += entry,
		n+=1

		df = {}
		for file in FILES:
			df[file] = pd.read_csv('%s/%s' %(results_dir,file), sep=',', header=None, index_col=[0,1,2])

		result = pd.concat([df[x] for x in FILES], axis=1)
		result.to_csv('%s/test_merged_mappers_all_%s_%s_%s_%s.csv' %(results_dir,epsilon,model,noise_model,n), header=False)
'''
