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