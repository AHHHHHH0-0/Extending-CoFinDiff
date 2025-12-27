"""
Embeddings module for CoFinDiff.
- Time embedding
- Macro embedding
"""

from .time_steps import timestep_embedding
from .macro_var import MacroEncoder

__all__ = [
    "timestep_embedding",
    "MacroEncoder",
]