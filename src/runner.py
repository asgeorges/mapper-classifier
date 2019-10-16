#!python
import sys, os, glob, shutil

sys.path.append('../lib/')

from mapper_tools import *
from pymapper import pymapper
from adversary import *
from predictor import *
from joiner import *
from variational_autoencoder import *
from os import listdir
from os.path import isfile, join
import time
import sys
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import Queue
from concurrent.futures import ProcessPoolExecutor

kpca = sys.argv[1]
real_matrix = sys.argv[2]
num_int = int(sys.argv[3])
num_bin = int(sys.argv[4])
classifier = sys.argv[5]
directory = sys.argv[6]
# filter_type = sys.argv[7]
map_method = sys.argv[7]
file_train = sys.argv[8]

stamp = '%s' % (file_train.split('.')[0])
total_range = range(1, 3, 1)
filter_type = None


def initial_mapper(argsl):
    skip_ = -1

    k = argsl[0]
    kpca = argsl[1]
    real_matrix = argsl[2]
    num_int = argsl[3]
    num_bin = argsl[4]
    classifier = argsl[5]
    directory = argsl[6]
    map_method = argsl[7]
    file_train = argsl[8]
    # try:
    pymapper(file_train, 'trueexamples_in.csv', '../data_train_fashion_unnormalized/', '../data_test_fashion_unnormalized',
             train='yes', kpca=kpca, real_matrix=real_matrix, map_method=map_method, dimensions=1, component=k,
             num_int=num_int, num_bin=num_bin, idx='yes', filter_type=filter_type)
    pymapper(file_train, 'trueexamples_in.csv', '../data_train_fashion_unnormalized/', '../data_test_fashion_unnormalized',
             train='no', kpca=kpca, real_matrix=real_matrix, map_method=map_method, dimensions=1, component=k,
             num_int=num_int, num_bin=num_bin, idx='yes', filter_type=filter_type)
    # except:
    #	skip_ = k

    return skip_


###########################################################################
# Start run with the initial training & testing via pool
###########################################################################
ARGS = [kpca, real_matrix, num_int, num_bin, classifier, directory, map_method, file_train]
argsl = [[k] + ARGS for k in total_range]

PROCS = 4
pool = Pool(PROCS)
skip = pool.map(initial_mapper, argsl, chunksize=1)
skip = [x for x in skip if x > -1]
print(skip)
pool.close()

skip = []
PCArange = [x for x in total_range if x not in skip]
total_merged = len(PCArange)
###########################################################################
# Do Processing
###########################################################################
joiner('../results', num_int, num_bin, PCArange, stamp)
bash_command('mv ../results/*csv ../results/%s/' % (directory))
time.sleep(15)
bash_command('cp ../results/%s/matrix* ../results/' % (directory))

###########################################################################
# Train up RF or NN classifier for prediction
###########################################################################
time.sleep(120)
predictor('merged_mappers_all_%s.csv' % (total_merged), 'test_merged_mappers_all_%s.csv' % (total_merged),
          '../results/%s' % (directory), method=classifier)

###########################################################################
# Run the adversarial scheme to construct robustness
###########################################################################

adversary('../results/%s' % (directory), 'test_merged_mappers_all_%s.csv' % (total_merged), num_int, num_bin, PCArange,
          file_train,
          real_matrix=real_matrix, noise_model='gaussian', classification_model=classifier, kpca=kpca,
          filter_type=filter_type, map_method=map_method)

###########################################################################
# Do processing
###########################################################################
# time.sleep(15)
# bash_command('rm ../data_temp/*')
# bash_command('rm ../results/*csv')
# time.sleep(300)
