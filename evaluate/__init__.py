"""
Evaluation functions for time series data.
"""
# Stylized facts
from .fisher_kurtosis import fisher_kurtosis
from .autocorrelation import acf_squared
from .euclidean import summarize_distances, sample_pair_indices, euclidean_on_pairs
from .dtw import dtw_on_pairs
from .micro_cond import adherence_table

__all__ = [
    'fisher_kurtosis',
    'acf_squared',
    'summarize_distances',
    'sample_pair_indices',
    'euclidean_on_pairs',
    'dtw_on_pairs',
    'adherence_table',
]
