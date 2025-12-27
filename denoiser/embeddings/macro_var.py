import torch
import torch.nn as nn

from config import config


class MacroEncoder(nn.Module):
    """
    This MLP embeds raw macroeconomic features for FiLM modulation.
    """
    
    def __init__(
        self,
        input_dim: int = config.MACRO_INPUT_DIM,
        hidden_dim: int = config.MACRO_HIDDEN_DIM,
        output_dim: int = config.MACRO_OUTPUT_DIM
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, macro: torch.Tensor) -> torch.Tensor:
        return self.net(macro)

