"""
Euclidean diversity utility functions.
"""

import numpy as np


def summarize_distances(values):
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p05': float(np.percentile(values, 5)),
        'p95': float(np.percentile(values, 95)),
        'n_pairs': int(values.shape[0]),
    }


def sample_pair_indices(n_series, max_pairs, rng):
    total_pairs = n_series * (n_series - 1) // 2
    target = min(max_pairs, total_pairs)

    pairs = set()
    while len(pairs) < target:
        i = int(rng.integers(0, n_series))
        j = int(rng.integers(0, n_series))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))

    pair_arr = np.array(list(pairs), dtype=np.int64)
    return pair_arr[:, 0], pair_arr[:, 1]


def euclidean_on_pairs(samples, idx_i, idx_j):
    diff = samples[idx_i] - samples[idx_j]
    return np.sqrt(np.sum(diff * diff, axis=1))
    