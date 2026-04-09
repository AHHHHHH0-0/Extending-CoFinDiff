"""
Configuration for the project.
"""

import os
import torch

# Project paths (relative paths from project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Core settings
SEED = 3407
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
