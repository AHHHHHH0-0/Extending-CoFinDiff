"""
Utility functions.
"""

import math
import torch
import torch.nn.functional as F

from config.config import config


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


def get_beta_schedule(
    schedule: str = config.BETA_SCHEDULE,
    timesteps: int = config.TIMESTEPS,
    beta_start: float = config.BETA_START,
    beta_end: float = config.BETA_END,
    device: str = config.DEVICE
) -> torch.Tensor:
    """
    Get noise schedule (beta values) for diffusion.
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, device=device)
    elif schedule == "quadratic":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, device=device) ** 2
    elif schedule == "cosine":
        steps = timesteps + 1
        s = 0.008
        t = torch.linspace(0, timesteps, steps, device=device)
        alphas_bar = torch.cos((t / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
