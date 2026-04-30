"""
PyTorch Dataset for loading preprocessed financial data from JSON.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Dict, List


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

    def get_subset(self, indices: List[int]) -> 'Subset':
        return Subset(self, indices)

    def compute_norm_stats(self) -> Dict[str, list]:
        """
        Compute per-condition mean and std from the full dataset (4 or 2 conditions).
        """
        trends = np.array([s['trend'] for s in self.data], dtype=np.float64)
        rvs = np.array([s['realized_vol'] for s in self.data], dtype=np.float64)
        irs = np.array([s['interest_rate'] for s in self.data], dtype=np.float64)
        vixs = np.array([s['volatility_index'] for s in self.data], dtype=np.float64)

        return {
            'cond_means': [float(trends.mean()), float(rvs.mean()), float(irs.mean()), float(vixs.mean())],
            'cond_stds': [float(trends.std()), float(rvs.std()), float(irs.std()), float(vixs.std())],
            'macro_means': [float(irs.mean()), float(vixs.mean())],
            'macro_stds': [float(irs.std()), float(vixs.std())]
        }
