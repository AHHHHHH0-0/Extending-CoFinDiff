"""
Create sinusoidal timestep embeddings for the denoiser.
"""

import torch
import torch.nn.functional as F
import math


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings for diffusion.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # Pad 1 for odd dimensions
    if dim % 2:
        emb = F.pad(emb, (0, 1))
        
    return emb
    