"""
Preprocess stock prices into wavelet coefficients.
"""

import numpy as np
import torch

from preprocessing.haar_wavelet import HaarWaveletTransform
from config import preprocess_config


def preprocess_prices(prices: np.ndarray, start:int) -> np.ndarray:
    """
    Preprocess 1D price series:
    1. Compute log returns: r_t = log(p_{t+1}) - log(p_t)
    2. Standardize to zero mean and unit variance
    3. Apply Haar wavelet transform
    """
    # Compute log returns
    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)
    
    # Take T timesteps starting from start
    log_returns_windowed = log_returns[start:start+preprocess_config.T]

    # Get trend and realized vol
    trend = _get_trend(log_returns_windowed)
    realized_vol = _get_realized_vol(log_returns_windowed)
    
    # Standardize
    mean = log_returns_windowed.mean()
    std = log_returns_windowed.std()
    standardized_returns = (log_returns_windowed - mean) / std
    
    # Convert to torch tensor for wavelet transform
    standardized_tensor = torch.from_numpy(standardized_returns).float()
    
    # Apply Haar wavelet transform
    wavelet_transform = HaarWaveletTransform()
    wavelet_coeffs = wavelet_transform(standardized_tensor)
    wavelet_coeffs = wavelet_coeffs.squeeze(0)
    
    return wavelet_coeffs, trend, realized_vol

def _get_trend(log_returns: np.ndarray) -> float:
    """
    Get the trend of the log returns.
    """
    log_returns = log_returns * 100
    trend = log_returns.sum()
    return trend

def _get_realized_vol(log_returns: np.ndarray) -> float:
    """
    Get the realized volatility of the wavelet coefficients.
    """
    log_returns = log_returns * 100
    realized_vol = (log_returns ** 2).sum()
    return realized_vol
