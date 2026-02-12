"""
Validation function for training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion import Diffusion
from config import project_config


def validate(
    denoiser: nn.Module,
    cond_encoder: nn.Module,
    diffusion: Diffusion,
    val_loader: DataLoader,
    device: str = project_config.DEVICE,
) -> float:
    """
    Compute validation loss.
    """
    denoiser.eval()
    cond_encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Get data and add channel dimension
            x0 = batch['returns_2d'].unsqueeze(1).to(device)  # (B, 1, H, W)
            B = x0.size(0)
            
            # Encode conditions
            cond_tokens = cond_encoder(
                trend=batch['trend'].to(device),
                realized_vol=batch['realized_vol'].to(device),
                interest_rate=batch['interest_rate'].to(device),
                volatility_index=batch['volatility_index'].to(device)
            )
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.T, (B,), device=device)
            
            # Compute loss
            loss = diffusion.loss(denoiser, x0, t, cond_tokens)
            total_loss += loss.item()
            num_batches += 1
    
    denoiser.train()
    cond_encoder.train()
    
    return total_loss / num_batches if num_batches > 0 else 0.0
