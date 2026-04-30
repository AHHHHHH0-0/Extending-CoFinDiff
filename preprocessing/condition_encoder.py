import torch
import torch.nn as nn

from config import preprocess_config


"""
Micro-economic condition encoder (augmented with macro conditions).
Concatenate 2 scalars -> FC -> 1D Conv -> 2D Conv -> Spatial tokens (B, H*W, cond_output_dim)
"""
class MicroConditionEncoder(nn.Module):
    def __init__(
        self,
        macro_means: list,
        macro_stds: list,
        num_micro_scalars: int = preprocess_config.NUM_MICRO_SCALARS,
        cond_fc_hidden_dim: int = preprocess_config.COND_FC_HIDDEN_DIM,
        cond_1d_channels: int = preprocess_config.COND_1D_CHANNELS,
        cond_2d_channels: int = preprocess_config.COND_2D_CHANNELS,
        cond_output_dim: int = preprocess_config.COND_OUTPUT_DIM,
        target_shape: tuple = preprocess_config.TARGET_SHAPE
    ):
        super().__init__()

        # Fixed z-score normalization for macro conditions (interest_rate, volatility_index)
        self.register_buffer('macro_norm_mean', torch.tensor(macro_means, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('macro_norm_std',  torch.tensor(macro_stds,  dtype=torch.float32).unsqueeze(0))

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
        return (macro_emb - self.macro_norm_mean) / self.macro_norm_std

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
Cross-attention encoder only.
- trend
- realized volatility
- interest rate
- volatility index
Concatenate 4 scalars → FC → 1D Conv → 2D Conv → Spatial representation (H, W, context_dim)
"""
class ConditionEncoder(nn.Module):
    def __init__(
        self,
        cond_means: list,
        cond_stds: list,
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

        # Fixed z-score normalization using stats computed from the training dataset
        self.register_buffer('norm_mean', torch.tensor(cond_means, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('norm_std',  torch.tensor(cond_stds,  dtype=torch.float32).unsqueeze(0))
        
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
        
        # Concatenate: (B, 4)
        x = torch.cat([trend, realized_vol, interest_rate, volatility_index], dim=1)
        x = (x - self.norm_mean) / self.norm_std
        
        # FC: (B, 4) → (B, H*W)
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
