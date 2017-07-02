# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# Visualizing 8x8 digits dataset (from sklearn). Visualizing first half, then transforming the rest.
# Feel free to play with starting_y

import sys
import os
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

# Importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import dynamic_tsne

if __name__ == "__main__":
    data = load_digits()
    X_all = data.images.reshape((-1, 64)) # 8x8 image to 64-length vector
    X = X_all[:len(X_all)//2, :]
    X2 = X_all[len(X_all)//2:, :]
    labels_all = data.target
    labels = labels_all[:len(labels_all)//2]
    labels2 = labels_all[len(labels_all)//2:]

    dTSNE = dynamic_tsne.DynamicTSNE(perplexity=20)
    y = dTSNE.fit(X, verbose=2, optimizer_kwargs={'momentum': 0.8}, random_seed=0)
    # Option 1. Start with random values
    # starting_y = None
    # Option 2. Start with Ys that corresponded to closest original Xs
    starting_y = 'closest'
    # Option 3. Start with a center of class. Only for testing, it won't be available in training.
    #starting_y = np.zeros((len(labels2), 2))
    #for i in range(len(starting_y)):
    #    starting_y[i,:] = np.mean(y[labels == labels2[i], :], axis=0)
    y2 = dTSNE.transform(X2, y=starting_y, verbose=2,
                         optimizer_kwargs={'momentum': 0.8, 'n_iter': 3000}, random_seed=1)

    color_list = ['blue', 'red', 'green', 'yellow', 'cyan', 'black', 'magenta', 'pink', 'brown', 'orange']

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1], c=color_list[l])
        legend_list.append(str(data.target_names[l]))
    for l in set(sorted(labels)):
        plt.scatter(y2[labels2 == l, 0], y2[labels2 == l, 1], c=color_list[l], marker='v')
        legend_list.append(str(data.target_names[l])+" embedded")
    # It did not work. Clusters were elsewhere.
    #plt.xlim([-70,90])
    #plt.ylim([-90,70])
    plt.legend(legend_list)
    plt.show()
