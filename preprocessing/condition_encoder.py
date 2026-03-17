"""
Micro-economic condition encoder for cross-attention conditioning (Trend, Realized Volatility).
Concatenate 2 scalars -> FC -> 1D Conv -> 2D Conv -> Spatial tokens (B, H*W, cond_output_dim)
"""

import torch
import torch.nn as nn

from config import preprocess_config


class MicroConditionEncoder(nn.Module):
    """
    Encode trend and realized volatility into spatial tokens for cross-attention conditioning.
    """
    
    def __init__(
        self,
        num_micro_scalars: int = preprocess_config.NUM_MICRO_SCALARS,
        cond_fc_hidden_dim: int = preprocess_config.COND_FC_HIDDEN_DIM,
        cond_1d_channels: int = preprocess_config.COND_1D_CHANNELS,
        cond_2d_channels: int = preprocess_config.COND_2D_CHANNELS,
        cond_output_dim: int = preprocess_config.COND_OUTPUT_DIM,
        target_shape: tuple = preprocess_config.TARGET_SHAPE
    ):
        super().__init__()
        
        self.H, self.W = target_shape
        self.spatial_size = self.H * self.W
        
        # FC layer: (B, 2) -> (B, H*W)
        self.fc = nn.Sequential(
            nn.Linear(num_micro_scalars, cond_fc_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_fc_hidden_dim, self.spatial_size)
        )
        
        # 1D Conv layer: (B, 1, H*W) -> (B, cond_1d_channels, H*W)
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=cond_1d_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(min(32, cond_1d_channels), cond_1d_channels),
            nn.SiLU()
        )
        
        # 2D Conv layer: (B, cond_1d_channels, H, W) -> (B, cond_output_dim, H, W)
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=cond_1d_channels,
                out_channels=cond_2d_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(min(32, cond_2d_channels), cond_2d_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=cond_2d_channels,
                out_channels=cond_output_dim,
                kernel_size=3,
                padding=1
            )
        )
    
    def forward(
        self,
        trend: torch.Tensor,
        realized_vol: torch.Tensor,
    ) -> torch.Tensor:

        B = trend.size(0)
        
        # Concatenate: (B, 2)
        x = torch.cat([trend, realized_vol], dim=1)
        
        # FC: (B, 2) -> (B, H*W)
        x = self.fc(x)
        
        # Reshape: (B, cond_fc_hidden_dim) → (B, 1, cond_fc_hidden_dim)
        x = x.unsqueeze(1)
        
        # 1D Conv: (B, 1, cond_fc_hidden_dim) → (B, cond_1d_channels, cond_fc_hidden_dim)
        x = self.conv1d(x)
        
        # Reshape: (B, cond_1d_channels, 256) → (B, cond_1d_channels, H, W)
        x = x.view(B, -1, self.H, self.W) 
        
        # 2D Conv: (B, cond_1d_channels, H, W) → (B, cond_output_dim, H, W)
        x = self.conv2d(x) 
        
        # Flatten spatial dimensions: (B, cond_output_dim, H, W) → (B, H*W, cond_output_dim)
        x = x.view(B, x.size(1), -1).transpose(1, 2)
        
        return x
