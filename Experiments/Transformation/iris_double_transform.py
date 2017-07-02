# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# Fitting half of iris dataset (from sklearn)
# Embedding other half using transform function.
# Feel free to play with transformation parameters.

import sys
import os
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import dynamic_tsne
import numpy as np

if __name__ == "__main__":
    data = load_iris()
    X = data.data
    labels = data.target

    dTSNE = dynamic_tsne.DynamicTSNE(perplexity=20)
    # Small dataset. Iterations are very fast, we can afford more
    y = dTSNE.fit(X, verbose=2, optimizer_kwargs={'momentum': 0.8, 'n_iter' : 3000}, random_seed=1)
    # Option 1. Start with random values
    # starting_y = None
    # Option 2. Start with Ys that corresponded to closest original Xs
    # starting_y = 'closest'
    # Option 3. Start with a center of class. Only for testing, it won't be available in training.
    #starting_y = np.zeros((len(labels), 2))
    #for i in range(len(starting_y)):
    #    starting_y[i,:] = np.mean(y[labels == labels[i], :], axis=0)
    # Option 4. Only for double-transform. Start with old values. Closest might mess up if X had duplicates.
    starting_y = y.copy()
    use_sigmas = np.concatenate((dTSNE.sigma, dTSNE.sigma), axis=0)
    y2 = dTSNE.transform(X, y=starting_y, verbose=2, use_sigmas=use_sigmas,
                         optimizer_kwargs={'momentum': 0.8, 'n_iter': 10000}, random_seed=1)
    print("Mean square error between y1 and y2: ", np.mean(np.sum((y-y2)**2, axis=1)))
    color_list = ['blue','orange','green']

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1], c=color_list[l])
        legend_list.append(str(data.target_names[l]))
    for l in set(sorted(labels)):
        plt.scatter(y2[labels == l, 0], y2[labels == l, 1], c=color_list[l], marker='v')
        legend_list.append(str(data.target_names[l])+" embedded")
    # TODO almost worked, but there were 3 outliers. Need to investigate.
    plt.xlim([-800, 1300])
    plt.ylim([-200, 200])
    plt.legend(legend_list)
    plt.show()
