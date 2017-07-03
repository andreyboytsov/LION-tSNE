# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# Fitting iris dataset (from sklearn). Generating embedding function and using it to visualize smooth transition from
# one value to another. Comparing to embedding_close, explores way more options, different transformations.

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

    # Embedding function requires unique arrays.
    # Well, embedding fuction can protect from it, but still we can get some confusion in mean square error, if it
    # "chooses" different sample
    un_array, un_idx = np.unique(X, axis=0, return_index=True)
    X = X[un_idx, :]
    labels = labels[un_idx]

    dTSNE = dynamic_tsne.DynamicTSNE(perplexity=20)
    # Small dataset. Iterations are very fast, we can afford more
    y = dTSNE.fit(X, verbose=2, optimizer_kwargs={'momentum': 0.8, 'n_iter' : 3000}, random_seed=1)
    embedder = dTSNE.generate_embedding_function(embedding_function_type='makeshift-lagrange-norm')
    embedder_weighted = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance')
    embedder_linear = dTSNE.generate_embedding_function(embedding_function_type='linear')
    start_index = 0
    end_index = 100
    steps = 100
    Xtransition = [X[start_index, :] + (X[end_index,:] - X[start_index, :])*i/steps for i in range(steps+1)]
    ytransition = [y[start_index] + (y[end_index] - y[start_index])*i/steps for i in range(steps+1)]
    y2 = embedder(Xtransition, verbose=2)
    y6 = embedder_weighted(Xtransition, verbose=2)
    y7 = embedder_linear(Xtransition, verbose=2)
    color_list = ['blue','orange','green']

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    legend_list.append(str(start_index)+" to "+str(end_index)+" transition (embedded)")
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted)")
    legend_list.append(str(start_index)+" to "+str(end_index)+" transition (linear)")
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1], c=color_list[l])
        legend_list.append(str(data.target_names[l]))
    plt.plot(y2[:, 0], y2[:, 1], color='black')
    plt.scatter(y2[:, 0], y2[:, 1], c='black',marker='x')
    plt.plot(y6[:, 0], y6[:, 1], color='cyan')
    plt.scatter(y6[:, 0], y6[:, 1], c='cyan',marker='x')
    plt.plot(y7[:, 0], y7[:, 1], color='grey')
    plt.scatter(y7[:, 0], y7[:, 1], c='grey',marker='x')
    plt.legend(legend_list)
    plt.xlim([np.min(y[:, 0])-20, np.max(y[:, 0])+20])
    plt.ylim([np.min(y[:, 1])-20, np.max(y[:, 1])+20])
    plt.title("Smooth transitioning of samples "+str(start_index) + " to " + str(end_index))
    plt.show()
