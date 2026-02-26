"""
Configuration for data preprocessing.
"""

# Haar wavelet parameters
HAAR_WAVELET_LEVELS = 6 # assert log2(T) == levels
TARGET_SHAPE = (8, 8) # assert H * W == T

# Data preprocessing parameters
T = 64 # power of 2 for the Haar wavelet transform
RETURN_SCALE_FACTOR = 100.0
STRIDE = 64

# Condition encoder parameters
NUM_CONDITION_SCALARS = 4
COND_FC_HIDDEN_DIM = 256
COND_1D_CHANNELS = 64
COND_2D_CHANNELS = 32
COND_OUTPUT_DIM = 32 # must match COND_CONTEXT_DIM in denoiser_config
