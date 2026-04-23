"""
DTW diversity utility functions.
"""

import numpy as np
from dtaidistance import dtw


def dtw_on_pairs(samples, idx_i, idx_j):
    out = np.empty(idx_i.shape[0], dtype=np.float64)
    for k in range(idx_i.shape[0]):
        out[k] = dtw.distance(samples[idx_i[k]], samples[idx_j[k]])
    return out
