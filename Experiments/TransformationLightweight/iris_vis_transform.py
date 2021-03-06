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
import lion_tsne_lightweight
import numpy as np
from sklearn.manifold import TSNE

if __name__ == "__main__":
    data = load_iris()
    np.random.seed(1)
    order = np.random.permutation(len(data.data))  # Shuffle the examples
    print(data.data.shape)
    all_data = data.data[order, :]
    all_labels = data.target[order]
    X = all_data[:len(all_data)//2]
    X_for_transformation = all_data[len(all_data)//2:]
    labels = all_labels[:len(all_labels)//2]
    labels2 = all_labels[len(all_labels)//2:]

    norm_TSNE = TSNE(perplexity=20)
    y = norm_TSNE.fit_transform(X)
    lwTSNE = lion_tsne_lightweight.LionTSNELightweight(X,y)

    embedder = lwTSNE.generate_lion_tsne_embedder(verbose=2, random_state=0, function_kwargs=
        {'y_safety_margin':0, 'radius_x_percentile':100, 'radius_y_percentile':100})
    y2 = embedder(X_for_transformation)

    color_list = ['blue','orange','green']

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1], c=color_list[l])
        legend_list.append(str(data.target_names[l]))
    for l in set(sorted(labels)):
        plt.scatter(y2[labels2 == l, 0], y2[labels2 == l, 1], c=color_list[l], marker='v')
        legend_list.append(str(data.target_names[l])+" embedded")
    # TODO almost worked, but there were 3 outliers. Need to investigate.
    #plt.xlim([-800, 1300])
    #plt.ylim([-100, 300])
    plt.legend(legend_list)
    plt.show()
