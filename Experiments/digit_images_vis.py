# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# Visualizing 8x8 digits dataset (from sklearn)

import sys
import os
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dynamic_tsne

if __name__ == "__main__":
    data = load_digits()
    X = data.images.reshape((-1, 64)) # 8x8 image to 64-length vector
    labels = data.target

    dTSNE = dynamic_tsne.DynamicTSNE(perplexity=20)
    y = dTSNE.fit(X, verbose=2, optimizer_kwargs={'momentum': 0.8}, random_seed=0)

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1])
        legend_list.append(str(data.target_names[l]))
    plt.legend(legend_list)
    plt.show()
