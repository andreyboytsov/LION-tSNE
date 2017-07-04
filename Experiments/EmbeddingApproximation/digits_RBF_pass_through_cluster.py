# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# # Visualizing 8x8 digits dataset (from sklearn). Generating embedding RBF function and using it to visualize smooth
# transition from one value to another. In transition straight Y line takes values through another cluster. However,
# it might be not the best option unless X is close to that cluster as well.

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
    embedder_rbf = dTSNE.generate_embedding_function(embedding_function_type='rbf')
    embedder_rbf_gaussian = dTSNE.generate_embedding_function(embedding_function_type='rbf',
                                                     function_kwargs={'function' : 'gaussian'})
    embedder_rbf_cubic = dTSNE.generate_embedding_function(embedding_function_type='rbf',
                                                     function_kwargs={'function' : 'cubic'})
    embedder_rbf_quintic = dTSNE.generate_embedding_function(embedding_function_type='rbf',
                                                     function_kwargs={'function' : 'quintic'})
    start_index = 500
    end_index = 600
    steps = 100
    Xtransition = [X[start_index, :] + (X[end_index,:] - X[start_index, :])*i/steps for i in range(steps+1)]
    y_rbf = embedder_rbf(Xtransition, verbose=2)
    y_rbf_gaussian = embedder_rbf_gaussian(Xtransition, verbose=2)
    y_rbf_cubic = embedder_rbf_cubic(Xtransition, verbose=2)
    y_rbf_quintic = embedder_rbf_quintic(Xtransition, verbose=2)

    plt.gcf().set_size_inches(10, 10)
    legend_list = list()
    legend_list.append("RBF (multiquadratic)")
    legend_list.append("RBF (Gaussian)")
    legend_list.append("RBF (cubic)")
    legend_list.append("RBF (quintic)")
    plt.scatter(y[:, 0], y[:, 1], c='gray')
    legend_list.append("Training data")
    plt.plot(y_rbf[:, 0], y_rbf[:, 1], color='red', marker='x')
    plt.plot(y_rbf_gaussian[:, 0], y_rbf_gaussian[:, 1], color='green', marker='x')
    plt.plot(y_rbf_cubic[:, 0], y_rbf_cubic[:, 1], color='blue', marker='x')
    plt.plot(y_rbf_quintic[:, 0], y_rbf_quintic[:, 1], color='black', marker='x')
    plt.legend(legend_list)
    plt.xlim([np.min(y[:, 0])-20, np.max(y[:, 0])+20])
    plt.ylim([np.min(y[:, 1])-20, np.max(y[:, 1])+20])
    plt.title("Smooth transitioning of samples "+str(start_index) + " to " + str(end_index))
    plt.show()
