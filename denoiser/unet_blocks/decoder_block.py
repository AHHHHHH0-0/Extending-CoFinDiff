import torch
import torch.nn as nn

from config import unet_config
from .sampling_blocks import Upsample
from .residual_block import ResBlock


class DecoderBlock(nn.Module):
    """
    Decoder block.
    - Upsampling Block
    - Concatenate skip connection
    - Residual Blocks
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_embed_dim: int = unet_config.TIME_EMBED_DIM,
        num_res_blocks: int = unet_config.NUM_RES_BLOCKS,
        upsample: bool = True
    ):
        super().__init__()
        
        # Upsampling
        self.has_upsample = upsample
        self.upsample = Upsample(in_channels) if upsample else nn.Identity()
        
        # ResBlocks (first one takes concatenated input)
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels + skip_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim
            )
            for i in range(num_res_blocks)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
    
        # Upsample
        h = self.upsample(x)
        
        # Concatenate skip connection
        h = torch.cat([h, skip], dim=1)
        
        # ResBlocks
        for block in self.res_blocks:
            h = block(h, time_emb)
        
        return h
