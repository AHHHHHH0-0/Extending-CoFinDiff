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

# Haar wavelet parameters
HAAR_WAVELET_LEVELS = 8 # assert log2(T) == levels
TARGET_SHAPE = (16, 16) # assert H * W == T

# Data preprocessing parameters
T = 256 # power of 2 for the Haar wavelet transform
RETURN_SCALE_FACTOR = 100.0

# Condition encoder parameters
COND_TOKEN_DIM = 20  
COND_HIDDEN_DIM = 64 
NUM_CONDITION_TOKENS = 4  # trend, realized_vol, interest_rate, market volatility index

# U-Net architecture parameters
UNET_IN_CHANNELS = 1
UNET_BASE_CHANNELS = 20  
UNET_CHANNEL_MULT = [1, 2, 4, 8]
UNET_NUM_RES_BLOCKS = 2  
UNET_KERNEL_SIZE = 3
UNET_NUM_GROUPS = 4  
UNET_DROPOUT = 0.0  
TIME_EMBED_DIM = 128  

# Cross-attention parameters
CROSS_ATTN_NUM_HEADS = 4
CROSS_ATTN_SCALE = 0.1 

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
EPOCHS = 300
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 20
EARLY_STOPPING = 100 
P_UNCOND = 0.1  
UPSAMPLE_EXTREME_EVENTS = 5 

# Sampling parameters
GUIDANCE_SCALE = 1.0 
RETURN_TRAJECTORY = False
