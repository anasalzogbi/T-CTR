from gensim.matutils import kullback_leibler, jaccard, hellinger
import numpy as np


def jensen_shannon_div(P, Q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions of equal length.

    ----- 
        :param P: Probability distributions that sum to 1
        :param Q: Probability distributions that sum to 1 
        :return: float
    """
    M = 0.5 * (P + Q)
    # return 0.5 * (_kldiv(P, M) +_kldiv(Q, M))
    return 0.5 * (kullback_leibler(P, M) + kullback_leibler(Q, M))


def _kldiv(A, B):
    return np.sum([v for v in A * np.log2(A / B) if not np.isnan(v)])
