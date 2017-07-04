# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# # Visualizing 8x8 digits dataset (from sklearn). Generating embedding function and using it to visualize smooth transition from
# one value to another. Comparing to embedding_smooth, explores just few options, only embeddings.

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
    temp = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, un_idx = np.unique(temp, return_index=True)
    X = X[un_idx, :]
    labels = labels[un_idx]

    dTSNE = dynamic_tsne.DynamicTSNE(perplexity=20)
    y = dTSNE.fit(X, verbose=2, optimizer_kwargs={'momentum': 0.8, 'n_iter': 1000}, random_seed=1)
    embedder2 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance')
    embedder3 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                  function_kwargs={'power' : 0.5})
    embedder4 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                  function_kwargs={'power' : 2})
    embedder5 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                  function_kwargs={'power' : 3})
    embedder6 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                  function_kwargs={'power' : 10})
    embedder7 = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                  function_kwargs={'power' : 20})
    # Regarding power behavior.
    # Imagine point 1st or 99th point. Distance to one of points is very small, 1/d is very large, hence the weight
    # is very large as well. If power<1 then for small distances (d**power) is larger than d, and 1/d**power is smaller
    # than d. Weight gets smaller, and last points moves further away (notice - for 0.5 all points are clumped).
    # For power > 1, for small distances (d**power) is smaller than d, and 1/d**power is larger, hence weights get
    # larger and second point moves closer to one of corner points.
    # Hence, too small power - all pts clumped in the middle, too high power - all points are closer to the ends,
    # middle is empty. Need to find balance.

    # It also seems low power created "pull to the center".
    start_index = 0
    end_index = 100
    steps = 100
    Xtransition = [X[start_index, :] + (X[end_index,:] - X[start_index, :])*i/steps for i in range(steps+1)]
    y3 = embedder2(Xtransition, verbose=2)
    y4 = embedder3(Xtransition, verbose=2)
    y5 = embedder4(Xtransition, verbose=2)
    y6 = embedder5(Xtransition, verbose=2)
    y7 = embedder6(Xtransition, verbose=2)
    y8 = embedder7(Xtransition, verbose=2)
    color_list = ['blue','orange','green']

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    color_list = ['blue', 'red', 'green', 'yellow', 'cyan', 'black', 'magenta', 'pink', 'brown', 'orange']
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted)")
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted, power 0.5)")
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted, power 2)")
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted, power 3)")
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted, power 10)")
    legend_list.append(str(start_index) + " to " + str(end_index) + " transition (weighted, power 20)")
    for l in set(sorted(labels)):
        plt.scatter(y[labels == l, 0], y[labels == l, 1], c=color_list[l])
        legend_list.append(str(data.target_names[l]))
    plt.plot(y3[:, 0], y3[:, 1], color='red')
    plt.scatter(y3[:, 0], y3[:, 1], c='red',marker='x')
    plt.plot(y4[:, 0], y4[:, 1], color='cyan')
    plt.scatter(y4[:, 0], y4[:, 1], c='cyan',marker='x')
    plt.plot(y5[:, 0], y5[:, 1], color='brown')
    plt.scatter(y5[:, 0], y5[:, 1], c='brown',marker='x')
    plt.plot(y6[:, 0], y6[:, 1], color='gray')
    plt.scatter(y6[:, 0], y6[:, 1], c='gray',marker='x')
    plt.plot(y7[:, 0], y7[:, 1], color='gray')
    plt.scatter(y7[:, 0], y7[:, 1], c='gray',marker='x')
    plt.plot(y8[:, 0], y8[:, 1], color='blue')
    plt.scatter(y8[:, 0], y8[:, 1], c='blue',marker='x')
    X_far = [100]*64
    y_far = embedder2(X_far)
    legend_list.append("Very far away X point will be put here")
    plt.scatter(y_far[:, 0], y_far[:, 1], c='black', marker = 'X', s=50)
    print("Y for far away X point: ", y_far)
    plt.legend(legend_list)
    plt.xlim([np.min(y[:, 0])-20, np.max(y[:, 0])+20])
    plt.ylim([np.min(y[:, 1])-20, np.max(y[:, 1])+20])
    plt.show()
