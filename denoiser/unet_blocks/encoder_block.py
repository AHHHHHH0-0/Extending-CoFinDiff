import torch
import torch.nn as nn

from config import unet_config
from .residual_block import ResBlock
from .sampling_blocks import Downsample


class EncoderBlock(nn.Module):
    """
    Encoder block.
    - Residual Blocks
    - Save skip connection
    - Downsampling Block
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int = unet_config.TIME_EMBED_DIM,
        num_res_blocks: int = unet_config.NUM_RES_BLOCKS,
        downsample: bool = True
    ):
        super().__init__()
        
        # ResBlocks
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim
            )
            for i in range(num_res_blocks)
        ])
        
        # Downsampling
        self.downsample = Downsample(out_channels) if downsample else nn.Identity()
        self.has_downsample = downsample
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> tuple:

        # ResBlocks
        h = x
        for block in self.res_blocks:
            h = block(h, time_emb)
        
        # Save skip connection
        skip = h
        
        # Downsample
        h = self.downsample(h)
        
        return h, skip