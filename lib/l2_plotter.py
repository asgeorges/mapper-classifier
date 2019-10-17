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


def l2_plotter(data_set, size, lower_bound, upper_bound, noise_model):


    noise_model = noise_model.replace(" ", "_")

    results_dir = '../results_final/%s/%s' %(data_set,size)
    stamp = '%s %s' %(size, str.capitalize(data_set))

    thresholds = np.linspace(lower_bound, upper_bound , 100)

    fig, ax = plt.subplots()


    if data_set=='mnist':
        if size=='10k':
            ######## MNIST, 10k
            architectures = ['PCA', 'CAE','DAE','VAE', 'CNN']
            c = ['blue', 'deepskyblue','cyan','slateblue', 'red']
            initial_accuracies = [9453, 9361, 9399, 9495, 9700]
        elif size=='60k':
            ######## MNIST 60k
            architectures = ['PCA', 'VAE', 'CNN']
            c = ['blue', 'slateblue', 'red']
            initial_accuracies = [9733,9768, 9851]
    elif data_set=='fashion':
        if size=='10k':
            ######## FASHION, 10k
            architectures=['PCA','CAE','VAE','CNN']
            c = ['blue', 'deepskyblue', 'slateblue','red']
            initial_accuracies=[8164,8127,8050,8935]
        elif size=='60k':
            ######## FASHION 60k
            architectures = ['PCA', 'VAE', 'CNN']
            c = ['blue', 'slateblue', 'red']
            initial_accuracies = [8672, 8757, 9336]

    accuracies = {}

    for n, architecture in enumerate(architectures):
        file = '%s/l2norms_mlp_%s_%s.csv' % (results_dir, str.lower(noise_model),architecture)
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

    plt.legend(loc="lower left", fontsize=12)
    plt.xlabel('$l^2$-Norm', fontsize=12)
    plt.ylabel('Normalized Accuracy', fontsize=12)
    ax.grid()
    # ax.set_yscale('log')
    noise_model = noise_model.replace("_", " ")
    plt.title("Normalized Classification Accuracy, %s, %s" % (stamp, noise_model), fontsize=12)
    noise_model = noise_model.replace(" ", "_")
    stamp = stamp.replace(" ", "_")
    plt.savefig('../results_final/Total_L2_Accuracy_%s_%s.pdf' % (stamp, noise_model), bbox_inches='tight',
                pad_inches=0)
    plt.show()
    plt.close()


if __name__ == '__main__':
    l2_plotter()
