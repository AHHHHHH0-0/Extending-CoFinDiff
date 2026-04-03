import torch
import torch.nn as nn

from config import denoiser_config


class FiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) for macro-economic conditioning.
    Modulates feature maps (h) with per-channel scale (gamma) and shift (beta).
    """
    
    def __init__(
        self,
        num_macro_scalars: int,
        channels: int,
        film_hidden_dim: int = denoiser_config.FILM_HIDDEN_DIM,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(num_macro_scalars, film_hidden_dim),
            nn.SiLU(),
            nn.Linear(film_hidden_dim, channels * 2),
        )
        
        # Initialize final layer so gamma=1, beta=0 (identity transform at init)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(
        self,
        h: torch.Tensor,
        macro_emb: torch.Tensor,
    ) -> torch.Tensor:
    
        # Predict gamma and beta: (B, 2*C)
        params = self.mlp(macro_emb)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, C)
        
        # Broadcast to spatial dims: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # FiLM: shift gamma by +1 so initialization gives identity (1*h + 0)
        return (1 + gamma) * h + beta
