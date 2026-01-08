"""
Train a single step of the model.
"""

import torch
import torch.nn as nn

from cofindiff.diffusion import Diffusion


def train_step(
    model: nn.Module,
    diffusion: Diffusion,
    x0: torch.Tensor,
    macro: torch.Tensor,
    macro_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    p_uncond: float = 0.1
) -> float:
    """
    Perform a single training step with classifier-free guidance dropout.
    
    Args:
        model: Denoiser model
        diffusion: Diffusion object
        x0: Clean data of shape (B, C, T)
        macro: Macro features of shape (B, macro_dim)
        macro_encoder: Macro encoder network
        optimizer: Optimizer
        p_uncond: Probability of dropping conditioning for CFG training
        
    Returns:
        Loss value (float)
    """
    B = x0.size(0)
    device = x0.device
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.T, (B,), device=device)
    
    # Encode macro features
    cond_emb = macro_encoder(macro)
    
    # Classifier-free guidance: randomly drop conditioning
    if p_uncond > 0:
        mask = (torch.rand(B, device=device) < p_uncond).float().unsqueeze(1)
        cond_emb = cond_emb * (1 - mask)
    
    # Compute loss
    optimizer.zero_grad()
    loss = diffusion.loss(model, x0, t, cond_emb)
    loss.backward()
    optimizer.step()
    
    return loss.item()
