"""
Configuration for the project.
"""

import os
import torch

# Project paths (relative paths from project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Core settings
SEED = 3407
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TCN parameters
KERNEL_SIZE = 3
NUM_GROUPS = 8

# Diffusion parameters
TIMESTEPS = 1000
OBJECTIVE = 'pred_noise'
BETA_SCHEDULE = 'linear'
BETA_START = 1e-4
BETA_END = 0.02
AUTO_NORMALIZE = False

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 500
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 20

# Sampling parameters
GUIDANCE_SCALE = 1.0
RETURN_TRAJECTORY = False
