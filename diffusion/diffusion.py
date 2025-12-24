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
        q(x_t | x_0) = N(x_t; sqrt(alphas_bar_t) * x_0, alphas_bar_t (1 - alphas_bar_t) * I)
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
        l = E[||ε - ε_θ(x_t, t, c)||²]
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
        guidance_scale: float = config.GUIDANCE_SCALE
    ) -> torch.Tensor:
        """
        Sample from the reverse diffusion process. 
        p(x_{t-1} | x_t) = N(x_{t-1}; μ, q_t² * I)
        """
        t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
        
        # Predict noise (conditioned and unconditioned)
        if guidance_scale != 1.0:
            eps_cond = model(x, t_batch, cond_emb)
            eps_uncond = model(x, t_batch, torch.zeros_like(cond_emb))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x, t_batch, cond_emb)
            
        # Coefficients
        alpha = self.alphas[t]
        alpha_bar = self.alphas_bar[t]
        beta = self.betas[t]
        
        # Compute mean
        mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - alpha_bar) * eps)
        
        # Add noise
        if t > 0:
            noise = torch.randn_like(x)
            std = torch.sqrt(beta)
            mean += std * noise
        
        return mean
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int,],
        cond_emb: torch.Tensor,
        guidance_scale: float = config.GUIDANCE_SCALE,
        return_trajectory: bool = config.RETURN_TRAJECTORY
    ) -> torch.Tensor:
        """
        Generate samples via the full reverse diffusion process.
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        trajectory = [x] if return_trajectory else None
        
        # Reverse diffusion
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond_emb, guidance_scale)
            if return_trajectory:
                trajectory.append(x)
                
        if return_trajectory:
            return x, trajectory
        return x
