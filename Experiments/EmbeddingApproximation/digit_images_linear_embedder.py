# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# # Visualizing 8x8 digits dataset (from sklearn). Generating linear embedding function and comapring it to weighted
# embedding function. Using it to visualize smooth transition from one value to another.

import sys
import os
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import dynamic_tsne
import numpy as np

if __name__ == "__main__":
    data = load_digits()
    X = data.images.reshape((-1, 64)) # 8x8 image to 64-length vector
    labels = data.target

    # Embedding function requires unique arrays.
    # Well, embedding fuction can protect from it, but still we can get some confusion in mean square error, if it
    # "chooses" different sample
    un_array, un_idx = np.unique(X, axis=0, return_index=True)
    X = X[un_idx, :]
    labels = labels[un_idx]

    dTSNE = dynamic_tsne.DynamicTSNE(perplexity=20)
    # Small dataset. Iterations are very fast, we can afford more
    y = dTSNE.fit(X, verbose=2, optimizer_kwargs={'momentum': 0.8, 'n_iter': 1000}, random_seed=1)
    embedder2 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance')
    embedder3 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                  function_kwargs={'power': 20})
    embedder4 = dTSNE.generate_embedding_function(embedding_function_type='linear')

    # It also seems low power created "pull to the center".
    start_index = 0
    end_index = 100
    steps = 100
    Xtransition = [X[start_index, :] + (X[end_index,:] - X[start_index, :])*i/steps for i in range(steps+1)]
    y2 = embedder2(Xtransition, verbose=2)
    y3 = embedder3(Xtransition, verbose=2)
    y4 = embedder4(Xtransition, verbose=2)
    color_list = ['blue','orange','green']

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    color_list = ['blue', 'red', 'green', 'yellow', 'cyan', 'black', 'magenta', 'pink', 'brown', 'orange']
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted, power 1)")
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted, power 20)")
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (linear)")
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1], c=color_list[l])
        legend_list.append(str(data.target_names[l]))

    X_far = [100]*64
    y_far = embedder4(X_far)
    print("Very far away X point will be put here: ", y_far)

    plt.plot(y2[:, 0], y2[:, 1], color='red')
    plt.scatter(y2[:, 0], y2[:, 1], c='red',marker='x')
    plt.plot(y3[:, 0], y3[:, 1], color='cyan')
    plt.scatter(y3[:, 0], y3[:, 1], c='cyan',marker='x')
    plt.plot(y4[:, 0], y4[:, 1], color='brown')
    plt.scatter(y4[:, 0], y4[:, 1], c='brown',marker='x')
    plt.legend(legend_list)
    plt.xlim([np.min(y[:, 0])-20, np.max(y[:, 0])+20])
    plt.ylim([np.min(y[:, 1])-20, np.max(y[:, 1])+20])
    plt.show()
