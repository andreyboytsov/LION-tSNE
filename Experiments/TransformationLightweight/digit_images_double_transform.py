# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# Visualizing 8x8 digits dataset (from sklearn).
# Then transforming the same data. Goal - get the same Ys on transformation.
# Transformation is done with lightweight LION-tSNE - transformer-only thing.

import sys
import os
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

# Importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import lion_tsne_lightweight
from sklearn.manifold import TSNE

if __name__ == "__main__":
    data = load_digits()
    X = data.images.reshape((-1, 64)) # 8x8 image to 64-length vector
    labels = data.target

    norm_TSNE = TSNE(perplexity=20)
    y = norm_TSNE.fit_transform(X)
    lwTSNE = lion_tsne_lightweight.LionTSNELightweight(X,y)

    embedder = lwTSNE.generate_lion_tsne_embedder(verbose=2, random_state=0, function_kwargs=
        {'y_safety_margin':0, 'radius_x_percentile':99, 'radius_y_percentile':100})
    y2 = embedder(X)
    print("Mean square error between y1 and y2: ", np.mean(np.sum((y-y2)**2, axis=1)))
    color_list = ['blue', 'red', 'green', 'yellow', 'cyan', 'black', 'magenta', 'pink', 'brown', 'orange']

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1], c=color_list[l])
        legend_list.append(str(data.target_names[l]))
    for l in set(sorted(labels)):
        plt.scatter(y2[labels == l, 0], y2[labels == l, 1], c=color_list[l], marker='v')
        legend_list.append(str(data.target_names[l])+" embedded")
    # It did not work. Clusters were elsewhere.
    #plt.xlim([-70,90])
    #plt.ylim([-90,70])
    plt.legend(legend_list)
    plt.show()
