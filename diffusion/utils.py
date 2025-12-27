"""
Utility functions.
- Get noise schedule (beta values) for diffusion.
"""

import math
import torch

from config.config import config


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
