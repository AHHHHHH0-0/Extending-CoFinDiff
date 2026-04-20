"""
Diffusion module for CoFinDiff.
- Forward diffusion process (adding noise)
- Reverse diffusion process (denoising)
- Training and sampling utilities
"""

from .diffusion_ca_film import DiffusionCAFilm
from .diffusion_ca import DiffusionCA

__all__ = [
    "DiffusionCAFilm",
    "DiffusionCA"
]