"""
Train a single step of the model.
"""

import torch
import torch.nn as nn
from typing import Dict

from diffusion import Diffusion
from config import training_config, project_config


def train_step(
    denoiser: nn.Module,
    diffusion: Diffusion,
    x: torch.Tensor,
    conditions: Dict[str, torch.Tensor],
    micro_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    p_uncond: float = training_config.P_UNCOND,
    device: str = project_config.DEVICE,
) -> float:
    """
    Single training step with classifier-free guidance dropout.
    """
    
    B = x.size(0)
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.T, (B,), device=device)
    
    # Encode micro conditions
    micro_cond_tokens = micro_encoder(
        trend=conditions['trend'],
        realized_vol=conditions['realized_vol'],
    )
    
    # Stack macro conditions for FiLM: (B, 2)
    macro_emb = torch.cat([
        conditions['interest_rate'],
        conditions['volatility_index'],
    ], dim=1)
    
    # Classifier-free guidance dropout (zero out both paths together)
    if p_uncond > 0:
        uncond_mask = torch.rand(B, device=device) < p_uncond
        micro_cond_tokens = micro_cond_tokens * (~uncond_mask).view(B, 1, 1).float()
        macro_emb = macro_emb * (~uncond_mask).view(B, 1).float()
    
    # Compute loss and backprop
    optimizer.zero_grad()
    loss = diffusion.loss(denoiser, x, t, micro_cond_tokens, macro_emb)
    loss.backward()
    optimizer.step()
    
    return loss.item()
