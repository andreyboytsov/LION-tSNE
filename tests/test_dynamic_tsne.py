from sklearn.datasets import load_iris
import numpy as np

# Importing from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dynamic_tsne


def test_init():
    """
    Test - initialize and do not crash.
    :return:
    """
    dynamic_tsne.DynamicTSNE(perplexity=20)


def test_random_seed():
    """
    Sets random seed and tests that results are the same.
    :return:
    """
    data = load_iris()
    X = data.data
    dTSNE = dynamic_tsne.DynamicTSNE(perplexity=20)
    # Small dataset. Iterations are very fast, we can afford more
    y = dTSNE.fit(X, optimizer_kwargs={'momentum': 0.8, 'n_iter' : 3000}, random_seed=1)
    y2 = dTSNE.fit(X, optimizer_kwargs={'momentum': 0.8, 'n_iter': 3000}, random_seed=1)
    assert (y == y2).all(), "Same random seed - different results"


def test_embedding_function_exact_polynomial():
    """
    Creates embedding function and makes sure that for known X it produces exact Y.
    :return:
    """
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
    embedder = dTSNE.generate_embedding_function()
    y2 = embedder(X, verbose=2)
    assert np.mean(np.sum((y-y2)**2, axis=1)) == 0, "Embedding was not exact"


def test_embedding_function_exact_polynomial():
    """
    Creates embedding function and makes sure that for known X it produces exact Y.
    :return:
    """
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
    for power in [1, 0.5, 2]:
        embedder = dTSNE.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                     function_kwargs={'power' : power})
        y2 = embedder(X, verbose=2)
        assert np.mean(np.sum((y-y2)**2, axis=1)) == 0, "Embedding was not exact"
