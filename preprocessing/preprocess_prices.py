"""
Preprocess stock prices into wavelet coefficients.
"""

import torch

from preprocessing.haar_wavelet import HaarWaveletTransform
from config import preprocess_config


def preprocess_prices(prices: torch.Tensor, start: int) -> tuple:
    """
    Preprocess 1D price series:
    1. Compute log returns: r_t = log(p_{t+1}) - log(p_t)
    2. Standardize to zero mean and unit variance
    3. Apply Haar wavelet transform
    """
    # Compute log returns
    log_prices = torch.log(prices)
    log_returns = torch.diff(log_prices)
    
    # Take T timesteps starting from start
    log_returns_windowed = log_returns[start:start+preprocess_config.T]

    # Get trend and realized vol
    trend = _get_trend(log_returns_windowed)
    realized_vol = _get_realized_vol(log_returns_windowed)
    
    # Standardize
    mean = log_returns_windowed.mean()
    std = log_returns_windowed.std()
    standardized_returns = (log_returns_windowed - mean) / std

    # Apply Haar wavelet transform
    wavelet_transform = HaarWaveletTransform()
    wavelet_coeffs = wavelet_transform(standardized_returns)
    wavelet_coeffs = wavelet_coeffs.squeeze(0)
    
    return wavelet_coeffs, trend, realized_vol

def _get_trend(log_returns: torch.Tensor) -> float:
    """
    Get the trend of the log returns.
    """
    log_returns = log_returns * 100
    trend = torch.sum(log_returns)
    return trend

def _get_realized_vol(log_returns: torch.Tensor) -> float:
    """
    Get the realized volatility of the wavelet coefficients.
    """
    log_returns = log_returns * 100
    realized_vol = torch.sum(log_returns ** 2)
    return realized_vol
