import torch
import torch.nn as nn

from config import config


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    - learned scale (gamma)
    - learned shift (beta)
    - formula: y = x * (1 + gamma) + beta
    """
    
    def __init__(
        self, 
        hidden_dim: int = config.FILM_HIDDEN_DIM, 
        cond_dim: int = config.FILM_COND_DIM
    ):
        super().__init__()
        self.net = nn.Linear(cond_dim, 2 * hidden_dim)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None]  # (B, C, 1)
        beta = beta[:, :, None]    # (B, C, 1)
        return x * (1 + gamma) + beta
