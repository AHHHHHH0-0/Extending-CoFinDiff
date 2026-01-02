import torch
import torch.nn as nn
import torch.nn.functional as F

from config import unet_config


class ResBlock(nn.Module):
    """
    Residual block for U-Net with timestep conditioning.
    
    Architecture:
        x -> Conv2D -> GroupNorm -> SiLU -> (+time_emb) -> Conv2D -> GroupNorm -> Dropout -> SiLU -> (+x)
    
    Follows standard DDPM ResBlock design with pre-activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        kernel_size: int,
        num_groups: int = unet_config.RES_BLOCK_NUM_GROUPS,
        dropout: float = unet_config.RES_BLOCK_DROPOUT
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
        
        # Timestep projection
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
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
        
        # Add timestep embedding
        time_proj = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_proj
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)
        
        # Residual connection
        return h + self.residual_proj(residual)
        