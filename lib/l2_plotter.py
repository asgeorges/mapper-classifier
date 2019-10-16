#!python

import os, string, sys, csv
import numpy as np
from numpy import sqrt, mean, median
import matplotlib.pyplot as plt


def file_opener(file):
    """
    This opens the recorded l2 norms csv

    :param file:  Called something like l2norms_mlp2_gaussian.csv
    :return: An array of the l2 values
    """
    with open(file) as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        data = list(reader)

    eps = [float(line[0]) for line in data]

    cutoffs = []
    for line in data:
        for entry in line[1:]:
            cutoffs += float(entry),

    cutoffs = np.array(cutoffs)
    return cutoffs


def l2_plotter():
    results_dir = '../results_final/fashion/60k'
    stamp = '60k Fashion'
    noise_model = 'S&P'
    thresholds = np.linspace(1, 4.5, 100)

    fig, ax = plt.subplots()

    ######## MNIST, 10k
    # architectures = ['PCA', 'CAE','DAE','VAE', 'CNN']
    # c = ['blue', 'deepskyblue','cyan','slateblue', 'red']
    # initial_accuracies = [9453, 9361, 9399, 9495, 9700]

    ######## MNIST 60k
    # architectures = ['PCA', 'VAE', 'CNN']
    # c = ['blue', 'slateblue', 'red']
    # initial_accuracies = [9733,9768, 9851]

    ######## FASHION, 10k
    # architectures=['PCA','CAE','VAE','CNN']
    # c = ['blue', 'deepskyblue', 'slateblue','red']
    # initial_accuracies=[8164,8127,8050,8935]

    ######## FASHION 60k
    architectures = ['PCA', 'VAE', 'CNN']
    c = ['blue', 'slateblue', 'red']
    initial_accuracies = [8672, 8757, 9336]

    accuracies = {}

    for n, architecture in enumerate(architectures):
        file = '%s/l2norms_mlp_s&p_%s.csv' % (results_dir, architecture)
        accuracies[(architecture, noise_model)] = []
        l2norms = file_opener(file)
        print(min(l2norms), max(l2norms))
        for threshold in thresholds:
            num_entries = [entry for entry in l2norms if entry <= threshold]
            num_entries = len(num_entries)
            initial_accuracy = initial_accuracies[n]
            ratio = 1 - float(num_entries) / float(initial_accuracy)
            accuracies[(architecture, noise_model)].append(ratio)
        plt.plot(thresholds, accuracies[(architecture, noise_model)], label='%s' % (architecture), color=c[n])

    plt.legend(loc="lower left")
    plt.xlabel('$l^2$-Norm')
    plt.ylabel('Normalized Accuracy')
    ax.grid()
    # ax.set_yscale('log')
    plt.title("Normalized Classification Accuracy, %s, %s" % (stamp, noise_model))
    plt.savefig('../results_final/Total_L2_Accuracy_%s_%s.pdf' % (stamp, noise_model), bbox_inches='tight',
                pad_inches=0)
    plt.show()
    plt.close()


if __name__ == '__main__':
    l2_plotter()
