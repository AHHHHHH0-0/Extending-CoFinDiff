"""
DDPM Diffusion forward (noising) and reverse (denoising) processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config.config import config
from .utils import get_beta_schedule


class Diffusion:
    """
    Denoising Diffusion Probabilistic Model (DDPM) implementation.
    - Forward diffusion process (adding noise)
    - Reverse diffusion process (denoising)
    - Training and sampling utilities
    """
    
    def __init__(
        self,
        timesteps: int = config.TIMESTEPS,
        beta_schedule: str = config.BETA_SCHEDULE,
        beta_start: float = config.BETA_START,
        beta_end: float = config.BETA_END,
        device: str = config.DEVICE
    ):
        self.T = timesteps
        self.device = device
        
        # Get betas and alphas
        self.betas = get_beta_schedule(
            schedule=beta_schedule,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from the forward diffusion process.
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        # Get coefficients for the batch
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar)[t][:, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_bar)[t][:, None, None]
        
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    
    def loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the training loss (MSE on noise prediction).
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        # Forward diffusion
        x_t = self.q_sample(x0, t, noise)
        
        # Predict noise
        pred = model(x_t, t, cond_emb)
        
        # MSE loss
        return F.mse_loss(pred, noise)
    
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        cond_emb: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t) - single denoising step.
        
        Args:
            model: Denoiser model
            x: Current noisy sample x_t of shape (B, C, T)
            t: Current timestep (scalar)
            cond_emb: Conditioning embedding of shape (B, cond_dim)
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            
        Returns:
            Denoised sample x_{t-1} of shape (B, C, T)
        """
        B = x.size(0)
        t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)
        
        # Predict noise
        if guidance_scale != 1.0:
            # Classifier-free guidance
            eps_cond = model(x, t_batch, cond_emb)
            eps_uncond = model(x, t_batch, torch.zeros_like(cond_emb))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x, t_batch, cond_emb)
            
        # Get coefficients
        alpha = self.alphas[t]
        alpha_bar = self.alphas_bar[t]
        beta = self.betas[t]
        
        # Compute mean
        # μ = (1/√α) * (x_t - β/√(1-α̅) * ε)
        mean = (1 / torch.sqrt(alpha)) * (
            x - beta / torch.sqrt(1 - alpha_bar) * eps
        )
        
        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(x)
            std = torch.sqrt(beta)
            return mean + std * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond_emb: torch.Tensor,
        guidance_scale: float = 1.0,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Generate samples via the full reverse diffusion process.
        
        Args:
            model: Denoiser model
            shape: Shape of samples to generate (B, C, T)
            cond_emb: Conditioning embedding of shape (B, cond_dim)
            guidance_scale: Classifier-free guidance scale
            return_trajectory: If True, return all intermediate samples
            
        Returns:
            Generated samples of shape (B, C, T)
            If return_trajectory=True, returns (samples, trajectory)
        """
        device = cond_emb.device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        trajectory = [x] if return_trajectory else None
        
        # Reverse diffusion
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond_emb, guidance_scale)
            if return_trajectory:
                trajectory.append(x)
                
        if return_trajectory:
            return x, trajectory
        return x


