"""
Train a single step of the model.
"""

import torch
import torch.nn as nn
from typing import Dict

from diffusion.diffusion import Diffusion
from config import training_config

def train_step(
    device: str,
    denoiser: nn.Module,
    diffusion: Diffusion,
    x: torch.Tensor,
    conditions: Dict[str, torch.Tensor],
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    p_uncond: float = training_config.P_UNCOND,
) -> float:
    """
    Single training step with classifier-free guidance dropout.
    """
    
    B = x.size(0)
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.T, (B,), device=device)
    
    # Encode conditions
    cond_tokens = cond_encoder(
        trend=conditions['trend'],
        realized_vol=conditions['realized_vol'],
        interest_rate=conditions['interest_rate'],
        volatility_index=conditions['volatility_index']
    )
    
    # Classifier-free guidance dropout
    if p_uncond > 0:
        uncond_mask = torch.rand(B, device=device) < p_uncond
        uncond_mask = uncond_mask.view(B, 1, 1)
        cond_tokens = cond_tokens * (~uncond_mask).float()
    
    # Compute loss and backprop
    optimizer.zero_grad()
    loss = diffusion.loss(denoiser, x, t, cond_tokens)
    loss.backward()
    optimizer.step()
    
    return loss.item()
