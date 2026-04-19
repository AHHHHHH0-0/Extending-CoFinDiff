"""
Fisher's excess kurtosis.

Formula:
    K = T(T+1)/((T-1)(T-2)(T-3)) * Σ((r_t - r̄)/s)^4 - 3(T-1)² / ((T-2)(T-3))
"""

import numpy as np


def fisher_kurtosis(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    mu = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True, ddof=1)
    z4 = ((X - mu) / s) ** 4
    return (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * z4.sum(axis=1) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
