import torch
import torch.nn as nn
from typing import List, Optional

from config import preprocess_config


"""
Micro-economic condition encoder for CA-FiLM model.
"""
class MicroConditionEncoder(nn.Module):
    def __init__(
        self,
        macro_means: Optional[List[float]] = None,
        macro_stds:  Optional[List[float]] = None,
        micro_means: Optional[List[float]] = None,
        micro_stds:  Optional[List[float]] = None,
        num_micro_scalars: int = preprocess_config.NUM_MICRO_SCALARS,
        cond_fc_hidden_dim: int = preprocess_config.COND_FC_HIDDEN_DIM,
        cond_1d_channels: int = preprocess_config.COND_1D_CHANNELS,
        cond_2d_channels: int = preprocess_config.COND_2D_CHANNELS,
        cond_output_dim: int = preprocess_config.COND_OUTPUT_DIM,
        target_shape: tuple = preprocess_config.TARGET_SHAPE
    ):
        super().__init__()

        # Micro normalization stats (trend, realized_vol)
        _micro_means = micro_means if micro_means is not None else [0.0] * num_micro_scalars
        _micro_stds  = micro_stds  if micro_stds  is not None else [1.0] * num_micro_scalars
        self.register_buffer('micro_norm_mean', torch.tensor(_micro_means, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('micro_norm_std',  torch.tensor(_micro_stds,  dtype=torch.float32).unsqueeze(0))

        # Macro normalization stats (interest_rate, volatility_index)
        _macro_means = macro_means if macro_means is not None else [0.0, 0.0]
        _macro_stds  = macro_stds  if macro_stds  is not None else [1.0, 1.0]
        self.register_buffer('macro_norm_mean', torch.tensor(_macro_means, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('macro_norm_std',  torch.tensor(_macro_stds,  dtype=torch.float32).unsqueeze(0))

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

    def normalize_macro(self, macro_emb: torch.Tensor) -> torch.Tensor:
        """Normalize raw [interest_rate, volatility_index] using training stats."""
        return (macro_emb - self.macro_norm_mean) / self.macro_norm_std

    def forward(
        self,
        trend: torch.Tensor,
        realized_vol: torch.Tensor,
    ) -> torch.Tensor:

        B = trend.size(0)

        # Concatenate and normalize micro inputs: (B, 2)
        x = torch.cat([trend, realized_vol], dim=1)
        x = (x - self.micro_norm_mean) / self.micro_norm_std

        # FC: (B, 2) -> (B, H*W)
        x = self.fc(x)

        # Reshape: (B, H*W) → (B, 1, H*W)
        x = x.unsqueeze(1)

        # 1D Conv: (B, 1, H*W) → (B, cond_1d_channels, H*W)
        x = self.conv1d(x)

        # Reshape: (B, cond_1d_channels, H*W) → (B, cond_1d_channels, H, W)
        x = x.view(B, -1, self.H, self.W)

        # 2D Conv: (B, cond_1d_channels, H, W) → (B, cond_output_dim, H, W)
        x = self.conv2d(x)

        # Flatten spatial dimensions: (B, cond_output_dim, H, W) → (B, H*W, cond_output_dim)
        x = x.view(B, x.size(1), -1).transpose(1, 2)

        return x


"""
Cross-attention encoder for CA model.
"""
class ConditionEncoder(nn.Module):
    def __init__(
        self,
        cond_means: Optional[List[float]] = None,
        cond_stds:  Optional[List[float]] = None,
        num_condition_scalars: int = preprocess_config.NUM_CONDITION_SCALARS,
        cond_fc_hidden_dim: int = preprocess_config.COND_FC_HIDDEN_DIM,
        cond_1d_channels: int = preprocess_config.COND_1D_CHANNELS,
        cond_2d_channels: int = preprocess_config.COND_2D_CHANNELS,
        cond_output_dim: int = preprocess_config.COND_OUTPUT_DIM,
        target_shape: tuple = preprocess_config.TARGET_SHAPE
    ):
        super().__init__()

        self.H, self.W = target_shape
        self.spatial_size = self.H * self.W

        # Fixed z-score normalization.
        _means = cond_means if cond_means is not None else [0.0] * num_condition_scalars
        _stds  = cond_stds  if cond_stds  is not None else [1.0] * num_condition_scalars
        self.register_buffer('norm_mean', torch.tensor(_means, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('norm_std',  torch.tensor(_stds,  dtype=torch.float32).unsqueeze(0))
        
        # FC layer
        self.fc = nn.Sequential(
            nn.Linear(num_condition_scalars, cond_fc_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_fc_hidden_dim, self.spatial_size)
        )
        
        # 1D Conv layer
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

        # 2D Conv layer
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
        interest_rate: torch.Tensor,
        volatility_index: torch.Tensor
    ) -> torch.Tensor:

        B = trend.size(0)

        # Concatenate and normalize: (B, 4)
        x = torch.cat([trend, realized_vol, interest_rate, volatility_index], dim=1)
        x = (x - self.norm_mean) / self.norm_std

        # FC: (B, 4) → (B, H*W)
        x = self.fc(x)

        # Reshape: (B, H*W) → (B, 1, H*W)
        x = x.unsqueeze(1)

        # 1D Conv: (B, 1, H*W) → (B, cond_1d_channels, H*W)
        x = self.conv1d(x)

        # Reshape: (B, cond_1d_channels, H*W) → (B, cond_1d_channels, H, W)
        x = x.view(B, -1, self.H, self.W)

        # 2D Conv: (B, cond_1d_channels, H, W) → (B, cond_output_dim, H, W)
        x = self.conv2d(x)

        # Flatten spatial dimensions: (B, cond_output_dim, H, W) → (B, H*W, cond_output_dim)
        x = x.view(B, x.size(1), -1).transpose(1, 2)

        return x
