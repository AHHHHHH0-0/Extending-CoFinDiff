import torch
import torch.nn as nn
import torch.nn.functional as F

from config import denoiser_config


class ResBlock(nn.Module):
    """
    Residual block for U-Net with timestep conditioning.
    
    Architecture:
        x -> Conv2D -> GroupNorm -> SiLU -> (+time_emb) -> Conv2D -> GroupNorm -> Dropout -> SiLU -> (+x)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        kernel_size: int = denoiser_config.RES_BLOCK_KERNEL_SIZE,
        num_groups: int = denoiser_config.RES_BLOCK_NUM_GROUPS,
        dropout: float = denoiser_config.RES_BLOCK_DROPOUT
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First convolution block
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        
        # Timestep projection: Linear + Conv 
        self.time_linear = nn.Linear(time_embed_dim, out_channels)
        self.time_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection (project if channels don't match)
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:

        # Save for residual
        residual = x
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Timestep embedding: reshape from (B, C) to (B, C, 1, 1) for broadcasting
        time_proj = self.time_linear(time_emb)  # (B, out_channels)
        time_proj = time_proj.unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        time_proj = self.time_conv(time_proj)  # (B, out_channels, 1, 1)
        h = h + time_proj  # Broadcasts across spatial dimensions
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)
        
        # Residual connection
        return h + self.residual_proj(residual)
        