# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# Fitting iris dataset (from sklearn). Generating embedding function and using it to embed the same data. It should be
# EXACTLY the same values.

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
    embedder2 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance')
    embedder3 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                  function_kwargs={'power' : 0.5})
    embedder4 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                  function_kwargs={'power' : 2})
    embedder5 = dTSNE.generate_embedding_function(embedding_function_type='linear')
    y2 = embedder(X, verbose=2)
    y3 = embedder2(X, verbose=2)
    y4 = embedder3(X, verbose=2)
    y5 = embedder4(X, verbose=2)
    y6 = embedder5(X, verbose=2)
    print("Mean square error between y1 and y2: ", np.mean(np.sum((y-y2)**2, axis=1)))
    print("Mean square error between y1 and y3: ", np.mean(np.sum((y-y3)**2, axis=1)))
    print("Mean square error between y1 and y4: ", np.mean(np.sum((y-y4)**2, axis=1)))
    print("Mean square error between y1 and y5: ", np.mean(np.sum((y-y5)**2, axis=1)))
    print("Mean square error between y1 and y6: ", np.mean(np.sum((y-y6) ** 2, axis=1)))
    color_list = ['blue','orange','green']

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1], c=color_list[l])
        legend_list.append(str(data.target_names[l]))
    for l in set(sorted(labels)):
        plt.scatter(y2[labels == l, 0], y2[labels == l, 1], c=color_list[l], marker='v')
        legend_list.append(str(data.target_names[l])+" embedded (polynmial)")
    for l in set(sorted(labels)):
        plt.scatter(y3[labels == l, 0], y3[labels == l, 1], c=color_list[l], marker='x')
        legend_list.append(str(data.target_names[l]) + " embedded (weights)")
    plt.legend(legend_list)
    plt.show()
