import torch
import torch.nn as nn
from typing import Dict

from diffusion import DiffusionCAFilm, DiffusionCA
from config import training_config, project_config


""" 
Single training step for CA-Film.
"""  
def train_step_ca_film(
    denoiser: nn.Module,
    diffusion: DiffusionCAFilm,
    x: torch.Tensor,
    conditions: Dict[str, torch.Tensor],
    micro_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    p_uncond: float = training_config.P_UNCOND,
    device: str = project_config.DEVICE,
) -> float:

    B = x.size(0)
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.T, (B,), device=device)
    
    # Encode micro conditions
    micro_cond_tokens = micro_encoder(
        trend=conditions['trend'],
        realized_vol=conditions['realized_vol'],
    )
    
    # Stack and normalize macro conditions: (B, 2)
    macro_emb = torch.cat([conditions['interest_rate'], conditions['volatility_index']], dim=1)
    macro_emb = micro_encoder.normalize_macro(macro_emb)

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


"""
Single training step for CA.
"""
def train_step_ca(
    denoiser: nn.Module,
    diffusion: DiffusionCA,
    x: torch.Tensor,
    conditions: Dict[str, torch.Tensor],
    cond_encoder: nn.Module,
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

    # Classifier-free guidance dropout
    if p_uncond > 0:
        keep = (~(torch.rand(B, device=device) < p_uncond)).float().unsqueeze(1)  # (B, 1)
        trend         = conditions['trend']         * keep
        realized_vol  = conditions['realized_vol']  * keep
        interest_rate = conditions['interest_rate'] * keep
        volatility_index = conditions['volatility_index'] * keep
    else:
        trend, realized_vol = conditions['trend'], conditions['realized_vol']
        interest_rate, volatility_index = conditions['interest_rate'], conditions['volatility_index']

    # Encode conditions
    cond_tokens = cond_encoder(
        trend=trend,
        realized_vol=realized_vol,
        interest_rate=interest_rate,
        volatility_index=volatility_index,
    )

    # Compute loss and backprop
    optimizer.zero_grad()
    loss = diffusion.loss(denoiser, x, t, cond_tokens)
    loss.backward()
    optimizer.step()

    return loss.item()
    