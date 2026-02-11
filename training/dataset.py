"""
PyTorch Dataset for loading preprocessed financial data from JSON.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict


class FinancialDataset(Dataset):
    """
    Dataset for loading preprocessed financial time series data.
    """
    
    def __init__(self, json_path: str):
        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Get dimensions
        self.num_assets = len(self.data)
        self.H = len(self.data[0]['returns_2d'])
        self.W = len(self.data[0]['returns_2d'][0])
        
        print(f"Loaded {self.num_assets} assets from {json_path}")
        print(f"Data shape: {self.H}x{self.W}")
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sample
        sample = self.data[idx]
        
        # Convert returns to tensor
        returns_2d = torch.tensor(sample['returns_2d'], dtype=torch.float32)
        
        # Convert conditions to tensors
        trend = torch.tensor([sample['trend']], dtype=torch.float32)
        realized_vol = torch.tensor([sample['realized_vol']], dtype=torch.float32)
        interest_rate = torch.tensor([sample['interest_rate']], dtype=torch.float32)
        volatility_index = torch.tensor([sample['volatility_index']], dtype=torch.float32)
        
        return {
            'returns_2d': returns_2d,
            'trend': trend,
            'realized_vol': realized_vol,
            'interest_rate': interest_rate,
            'volatility_index': volatility_index
        }
