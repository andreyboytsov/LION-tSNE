from sklearn.datasets import load_iris

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
