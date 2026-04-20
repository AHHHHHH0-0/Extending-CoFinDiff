"""
Validation function for training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion import DiffusionCAFilm
from config import project_config


def validate(
    denoiser: nn.Module,
    micro_encoder: nn.Module,
    diffusion: DiffusionCAFilm,
    val_loader: DataLoader,
    device: str = project_config.DEVICE,
) -> float:
    """
    Compute validation loss.
    """
    denoiser.eval()
    micro_encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Get data and add channel dimension
            x0 = batch['returns_2d'].unsqueeze(1).to(device)  # (B, 1, H, W)
            B = x0.size(0)
            
            # Encode micro conditions for cross-attention
            micro_cond_tokens = micro_encoder(
                trend=batch['trend'].to(device),
                realized_vol=batch['realized_vol'].to(device),
            )
            
            # Stack macro conditions for FiLM: (B, 2)
            macro_emb = torch.cat([
                batch['interest_rate'].to(device),
                batch['volatility_index'].to(device),
            ], dim=1)
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.T, (B,), device=device)
            
            # Compute loss
            loss = diffusion.loss(denoiser, x0, t, micro_cond_tokens, macro_emb)
            total_loss += loss.item()
            num_batches += 1
    
    denoiser.train()
    micro_encoder.train()
    
    return total_loss / num_batches if num_batches > 0 else 0.0
