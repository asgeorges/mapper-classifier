# !python
import sys, os, glob, shutil

sys.path.append('../lib/')

from mapper_tools import *
from pymapper import pymapper
from adversary import *
from predictor import *
from joiner import *
from l2_plotter import *
from os import listdir
from os.path import isfile, join
import time
import numpy as np
import matplotlib.pyplot as plt

pymapper('trueexamples_in.csv', 'trueexamples_in.csv', '../data_train_fashion_unnormalized',
         '../data_test_fashion_unnormalized', projection='pca', filter_type=None, train='yes', map_method='cluster4',
         dimensions=1, component=1, num_int=10, num_bin=10, idx='yes')
pymapper('trueexamples_in.csv', 'trueexamples_in.csv', '../data_train_fashion_unnormalized',
         '../data_test_fashion_unnormalized', projection='pca', filter_type=None, train='no', map_method='cluster4',
         dimensions=1, component=1, num_int=10, num_bin=10, idx='yes')
