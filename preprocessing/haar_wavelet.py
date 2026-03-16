"""
Haar wavelet transform utilities for converting 1D time series to 2D images.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple

from config import preprocess_config


class HaarWaveletTransform(nn.Module):
    """
    Haar wavelet transform module for preprocessing 1D financial time series into 2D images.
    """

    def __init__(
        self, 
        levels: int = preprocess_config.HAAR_WAVELET_LEVELS, 
        target_shape: Tuple[int, int] = preprocess_config.TARGET_SHAPE,
        time_steps: int = preprocess_config.T
    ):
        super().__init__()
        self.levels = levels
        self.target_shape = target_shape
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform 1D sequence to 2D image.
        """
        # Apply multi-level Haar transform
        x_haar = self._multilevel_haar_transform(x)
        
        # Reshape to 2D
        x_2d = x_haar.reshape(-1, self.target_shape[0], self.target_shape[1])
        
        return x_2d
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform 2D image back to 1D sequence.
        """
        # Reshape to 1D
        x_1d = x.reshape(-1, self.target_shape[0] * self.target_shape[1])
        
        # Apply inverse Haar transform
        x_reconstructed = self._inverse_multilevel_haar_transform(x_1d)
        
        return x_reconstructed
        
    def _haar_transform_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 1D Haar wavelet transform.
        """
        # Split into even and odd indices
        even = x[..., 0::2]
        odd = x[..., 1::2]
        
        # Approximation (low-pass) and detail (high-pass) coefficients
        sqrt_2 = math.sqrt(2)
        approx = (even + odd) / sqrt_2
        detail = (even - odd) / sqrt_2
        
        return torch.cat([approx, detail], dim=-1)
    
    def _inverse_haar_transform_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse 1D Haar wavelet transform.
        """
        half = x.shape[-1] // 2
        
        # Split into approximation and detail
        approx = x[..., :half]
        detail = x[..., half:]
        
        # Reconstruct even and odd
        sqrt_2 = math.sqrt(2)
        even = (approx + detail) / sqrt_2
        odd = (approx - detail) / sqrt_2
        
        # Interleave
        result = torch.zeros_like(x)
        result[..., 0::2] = even
        result[..., 1::2] = odd
        
        return result
    
    def _multilevel_haar_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-level 1D Haar wavelet decomposition.
        """
        result = x.clone()
        
        # Apply transform at each level
        for level in range(self.levels):
            # Only transform the approximation coefficients from previous level
            length = self.time_steps // (2 ** level)
            result[..., :length] = self._haar_transform_1d(result[..., :length])
        
        return result

    def _inverse_multilevel_haar_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse multi-level 1D Haar wavelet reconstruction.
        """
        result = x.clone()
        
        # Apply inverse transform at each level in reverse order
        for level in range(self.levels - 1, -1, -1):
            length = self.time_steps // (2 ** level)
            result[..., :length] = self._inverse_haar_transform_1d(result[..., :length])
        
        return result
    