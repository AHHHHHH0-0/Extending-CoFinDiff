"""
Diffusion module for CoFinDiff.
- Forward diffusion process (adding noise)
- Reverse diffusion process (denoising)
- Training and sampling utilities
"""

from .diffusion import Diffusion

__all__ = [
    "Diffusion",
]