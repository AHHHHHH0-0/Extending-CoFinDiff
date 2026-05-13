"""
Preprocessing utilities for financial time series data.
- Preprocess prices with log returns and global scaling.
- Haar wavelet transform for converting 1D time series to 2D images.
- Micro condition encoder for cross-attention conditioning.
- Condition encoder for cross-attention conditioning.
- Global std computation for dataset-level scale normalization.
"""

from .preprocess_prices import preprocess_prices
from .condition_encoder import MicroConditionEncoder, ConditionEncoder
from .haar_wavelet import HaarWaveletTransform
from .global_std import get_global_std, load_global_stats

__all__ = [
    'preprocess_prices',
    'HaarWaveletTransform',
    'MicroConditionEncoder',
    'ConditionEncoder',
    'get_global_std',
    'load_global_stats',
]
