"""
Condition encoder for cross-attention conditioning.
- trend
- realized volatility
- interest rate
- volatility index
"""

import torch
import torch.nn as nn

from config import preprocess_config


class ConditionEncoder(nn.Module):
    """
    Encode 4 scalar conditions into 4 tokens for cross-attention.
    """
    
    def __init__(
        self,
        hidden_dim: int = preprocess_config.COND_HIDDEN_DIM,
        token_dim: int = preprocess_config.COND_TOKEN_DIM
    ):
        super().__init__()
        self.trend_encoder = _ScalarToTokenEncoder(hidden_dim, token_dim)
        self.volatility_encoder = _ScalarToTokenEncoder(hidden_dim, token_dim)
        self.interest_encoder = _ScalarToTokenEncoder(hidden_dim, token_dim)
        self.volatility_index_encoder = _ScalarToTokenEncoder(hidden_dim, token_dim)
    
    def forward(
        self,
        trend: torch.Tensor,
        realized_vol: torch.Tensor,
        interest_rate: torch.Tensor,
        volatility_index: torch.Tensor
    ) -> torch.Tensor:

        return torch.stack([
            self.trend_encoder(trend),
            self.volatility_encoder(realized_vol),
            self.interest_encoder(interest_rate),
            self.volatility_index_encoder(volatility_index)
        ], dim=1)


class _ScalarToTokenEncoder(nn.Module):
    """
    Encode a single scalar value into a token vector.
    
    Architecture:
        scalar → Linear(1, hidden) → SiLU → Linear(hidden, token_dim) → token
    """
    
    def __init__(self, hidden_dim: int, token_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
