import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


class CausalConv1d(nn.Conv1d):
    """
    Causal 1D convolution with dilation support, left padding for causality.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        **kwargs
    ):
        # Remove padding from kwargs
        kwargs.pop('padding', None)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            **kwargs
        )
        # Left padding 
        self.left_padding = (kernel_size - 1) * dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left padding 
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


