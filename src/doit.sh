#!/bin/bash

#arg1 is kpca
#arg2 yes is real matrix

python runner.py 'no' 'no' 10 10 'mlp2' 'fashion_pca' 'cluster4' 'trueexamples_in.csv' > \
../results_final/fashion_pca.txt