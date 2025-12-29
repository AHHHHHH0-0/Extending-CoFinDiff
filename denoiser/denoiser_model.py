import torch
import torch.nn as nn

from .embeddings import timestep_embedding
from .film import FiLM
from .tcn_block import TCNResidualBlock
from config import config


class CoFinDiffTCN(nn.Module):
    """
    CoFinDiff Denoiser class.
    """
    
    def __init__(
        self,
        in_channels: int = config.IN_CHANNELS,
        hidden_channels: int = config.HIDDEN_CHANNELS,
        num_layers: int = config.NUM_LAYERS,
        time_dim: int = config.TIME_DIM,
        cond_dim: int = config.COND_DIM,
        kernel_size: int = config.KERNEL_SIZE,
        num_groups: int = config.NUM_GROUPS
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        
        # Input projection: map input channels to hidden channels
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # TCN blocks with exponentially increasing dilation
        self.dilations = [2 ** i for i in range(num_layers)]
        
        self.blocks = nn.ModuleList([
            TCNResidualBlock(
                channels=hidden_channels,
                dilation=d,
                time_dim=time_dim,
                cond_dim=cond_dim,
                kernel_size=kernel_size,
                num_groups=num_groups
            )
            for d in self.dilations
        ])
        
        # Output projection: map hidden channels back to input channels
        self.output_proj = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        
        # Initialize output projection to zero for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the noise added to the input.
        """
        # Create and process timestep embedding
        t_emb = timestep_embedding(t, self.time_dim)  # (B, time_dim)
        t_emb = self.time_mlp(t_emb)  # (B, time_dim)
        
        # Project input to hidden dimension
        h = self.input_proj(x)  # (B, hidden_channels, T)
        
        # Pass through TCN blocks
        for block in self.blocks:
            h = block(h, t_emb, cond_emb)
            
        # Project back to input dimension
        return self.output_proj(h)  # (B, in_channels, T)
    
    def get_receptive_field(self) -> int:
        """
        Sanity check: calculate the receptive field of TCN.
        """
        rf = 1
        for d in self.dilations:
            # Each block has 2 causal convolutions
            rf += (self.kernel_size - 1) * d * 2
            
        return rf


