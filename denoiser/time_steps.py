import torch
import torch.nn.functional as F
import math


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    # Pad 1 for odd dimensions
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    
    return embedding
    