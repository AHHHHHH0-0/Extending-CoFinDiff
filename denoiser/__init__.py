"""
This module contains the TCN-based denoiser network and related components
for the conditional financial diffusion model.
- DDPM (Denoising Diffusion Probabilistic Models)
- TCN (Temporal Convolutional Network) denoiser
- Causal dilated convolutions
- Residual blocks with FiLM conditioning
- MacroEncoder for conditioning
"""

from .denoiser_model import CoFinDiffTCN
from .embeddings.macro_var import MacroEncoder

__all__ = [
    "CoFinDiffTCN", 
    "MacroEncoder"
]