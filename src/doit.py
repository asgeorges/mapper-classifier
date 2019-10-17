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



l2_plotter('mnist', '10k', 1, 5, 'Gauss Blur')
l2_plotter('mnist', '10k', 2, 5, 'Gaussian')
l2_plotter('mnist', '10k', 1, 5, 'S&P')

l2_plotter('mnist', '60k', 1, 5, 'Gauss Blur')
l2_plotter('mnist', '60k', 2, 5, 'Gaussian')
l2_plotter('mnist', '60k', 1, 5, 'S&P')


l2_plotter('fashion', '10k', 0, 4, 'Gauss Blur')
l2_plotter('fashion', '10k', 2, 5, 'Gaussian')
l2_plotter('fashion', '10k', 1, 4.5, 'S&P')

l2_plotter('fashion', '60k', 0, 4, 'Gauss Blur')
l2_plotter('fashion', '60k', 2, 5, 'Gaussian')
l2_plotter('fashion', '60k', 1, 4.5, 'S&P')

