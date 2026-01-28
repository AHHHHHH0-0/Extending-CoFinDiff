"""
Generate samples from the diffusion model with cross-attention conditioning.
"""

import torch
import torch.nn as nn

from diffusion.diffusion import Diffusion
from preprocessing import HaarWaveletTransform


@torch.no_grad()
def diffusion_generate(
    model: nn.Module,
    diffusion: Diffusion,
    cond_encoder: nn.Module,
    haar_transform: HaarWaveletTransform,
    trend: float,
    realized_vol: float,
    interest_rate: float,
    volatility_index: float,
    num_samples: int,
    num_channels: int,
    img_shape: tuple,
    guidance_scale: float,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generate samples with arbitrary user-specified conditioning values.
    """
    # Create condition tensors
    conditions = {
        'trend': torch.full((num_samples,), trend, device=device),
        'realized_vol': torch.full((num_samples,), realized_vol, device=device),
        'interest_rate': torch.full((num_samples,), interest_rate, device=device),
        'volatility_index': torch.full((num_samples,), volatility_index, device=device)
    }
    
    return _run_diffusion(
        model=model,
        diffusion=diffusion,
        cond_encoder=cond_encoder,
        haar_transform=haar_transform,
        conditions=conditions,
        num_channels=num_channels,
        img_shape=img_shape,
        guidance_scale=guidance_scale
    )


@torch.no_grad()
def _run_diffusion(
    model: nn.Module,
    diffusion: Diffusion,
    cond_encoder: nn.Module,
    haar_transform: HaarWaveletTransform,
    conditions: dict,
    num_channels: int,
    img_shape: tuple,
    guidance_scale: float
) -> torch.Tensor:
    """
    Generate financial time series samples using diffusion model.
    """
    # Set models to evaluation mode
    model.eval()
    cond_encoder.eval()
    
    # Set shape
    B = conditions['trend'].size(0)
    H, W = img_shape
    shape = (B, num_channels, H, W)
    
    # Encode conditions to tokens
    cond_tokens = cond_encoder(
        trend=conditions['trend'],
        realized_vol=conditions['realized_vol'],
        interest_rate=conditions['interest_rate'],
        volatility_index=conditions['volatility_index']
    )
    
    # Sample
    samples_2d = diffusion.sample(model, shape, cond_tokens, guidance_scale)
    
    # Inverse Haar transform
    samples_1d = haar_transform.inverse(samples_2d)
    
    return samples_1d



