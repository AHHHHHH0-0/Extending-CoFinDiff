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
    
    # Standardize
    mean = log_returns_windowed.mean()
    std = log_returns_windowed.std()
    standardized_returns = (log_returns_windowed - mean) / std
    
    # Convert to torch tensor for wavelet transform
    standardized_tensor = torch.from_numpy(standardized_returns).float()
    standardized_tensor = standardized_tensor.unsqueeze(0)
    
    # Apply Haar wavelet transform
    wavelet_transform = HaarWaveletTransform()
    wavelet_coeffs = wavelet_transform(standardized_tensor)
    
    return wavelet_coeffs
