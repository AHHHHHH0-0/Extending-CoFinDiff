"""
Generate samples from the model.
"""

import torch
import torch.nn as nn

from cofindiff.diffusion import Diffusion


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    diffusion: Diffusion,
    macro: torch.Tensor,
    macro_encoder: nn.Module,
    seq_length: int,
    num_channels: int,
    guidance_scale: float = 3.0
) -> torch.Tensor:
    """
    Generate financial time series samples.
    
    Args:
        model: Trained denoiser model
        diffusion: Diffusion object
        macro: Macro features of shape (B, macro_dim)
        macro_encoder: Macro encoder network
        seq_length: Length of sequences to generate
        num_channels: Number of channels/assets
        guidance_scale: Classifier-free guidance scale
        
    Returns:
        Generated samples of shape (B, num_channels, seq_length)
    """
    model.eval()
    macro_encoder.eval()
    
    B = macro.size(0)
    shape = (B, num_channels, seq_length)
    
    cond_emb = macro_encoder(macro)
    samples = diffusion.sample(model, shape, cond_emb, guidance_scale)
    
    return samples
