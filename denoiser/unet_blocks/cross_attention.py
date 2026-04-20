import torch
import torch.nn as nn
import torch.nn.functional as F

from config import denoiser_config


class SpatialCrossAttention(nn.Module):
    """
    Cross-attention module for spatial feature maps.
    - Flattens spatial dimensions
    - Applies cross-attention
    - Reshapes back
    """
    
    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int,
        scale: float = denoiser_config.CROSS_ATTN_SCALE
    ):
        super().__init__()

        # Group normalization before attention
        self.norm = nn.GroupNorm(
            num_groups=min(32, channels),
            num_channels=channels,
            eps=1e-6, 
        )
        
        # Cross-attention layer
        self.attn = _MultiHeadCrossAttention(
            query_dim=channels,
            context_dim=context_dim,
            num_heads=num_heads,
            scale=scale
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:

        B, C, H, W = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x_flat = x_norm.reshape(B, C, H * W).transpose(1, 2)
        
        # Apply cross-attention
        out = self.attn(x_flat, context)  # (B, H*W, C)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class _MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention.
    
    Architecture:
        Q = bottleneck features (spatial tokens)
        K, V = condition tokens
        Output = MultiHeadAttention(Q, K, V) with residual connection
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        scale: float
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.scale_factor = scale
        
        # Compute head dimension
        assert query_dim % num_heads == 0, \
            f"Query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = query_dim // num_heads

        # Inner dimension 
        inner_dim = num_heads * self.head_dim
        
        # Linear projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        # Scaling for attention scores
        self.attn_scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:

        B, N = x.shape[:2]
        M = context.shape[1]
        
        # Project to Q, K, V
        q = self.to_q(x)  # (B, N, inner_dim)
        k = self.to_k(context)  # (B, M, inner_dim)
        v = self.to_v(context)  # (B, M, inner_dim)
        
        # Reshape for multi-head attention
        # (B, N, inner_dim) -> (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, M) -> (B, num_heads, N, M)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        # (B, num_heads, N, M) @ (B, num_heads, M, head_dim) -> (B, num_heads, N, head_dim)
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        # (B, num_heads, N, head_dim) -> (B, N, inner_dim)
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        
        # Output projection
        out = self.to_out(out)  # (B, N, query_dim)
        
        # Scale to prevent over-conditioning
        out = out * self.scale_factor
        
        # Residual connection
        return x + out