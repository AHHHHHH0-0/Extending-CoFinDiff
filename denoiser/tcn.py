import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from .conv1d import CausalConv1d
from .layers import FiLM


class TCNResidualBlock(nn.Module):
    """
    One single Temporal Convolutional Network (TCN) residual block with conditioning.
    - Causal dilated convolution for temporal modeling
    - Group normalization for stability
    - Timestep embedding
    - Conditioning via FiLM 

    Architecture:
        x -> Conv1 -> Norm1 -> SiLU -> (+t_emb) -> Conv2 -> Norm2 -> FiLM -> SiLU -> (+x)
    """
    
    def __init__(
        self,
        channels: int,
        dilation: int,
        time_dim: int,
        cond_dim: int,
        kernel_size: int = config.KERNEL_SIZE,
        num_groups: int = config.NUM_GROUPS
    ):
        super().__init__()
        
        # First causal convolution
        self.conv1 = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        
        # Second causal convolution
        self.conv2 = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        
        # Group normalization layers
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        
        # Timestep projection
        self.time_proj = nn.Linear(time_dim, channels)
        
        # FiLM conditioning
        self.film = FiLM(channels, cond_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond_emb: torch.Tensor
    ) -> torch.Tensor:
        # First convolution block + group normalization + SiLU
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add timestep embedding
        h += self.time_proj(t_emb)[:, :, None]
        
        # Second convolution block + group normalization + SiLU (conditioning)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film(h, cond_emb) # conditioning
        h = F.silu(h)
        
        return x + h

