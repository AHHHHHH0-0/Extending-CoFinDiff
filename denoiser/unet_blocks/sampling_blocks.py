import torch
import torch.nn as nn
import torch.nn.functional as F

from config import denoiser_config


class Downsample(nn.Module):
    """
    Downsampling layer using strided convolution.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=denoiser_config.DOWN_SAMPLE_KERNEL_SIZE,
            stride=denoiser_config.DOWN_SAMPLE_STRIDE,
            padding=denoiser_config.DOWN_SAMPLE_PADDING
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer using nearest neighbor interpolation and convolution.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=denoiser_config.UP_SAMPLE_KERNEL_SIZE,
            padding=denoiser_config.UP_SAMPLE_PADDING
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
