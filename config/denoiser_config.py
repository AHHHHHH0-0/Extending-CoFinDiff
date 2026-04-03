"""
Configuration for the U-Net denoiser.
"""

# Down and up sampling blocks parameters
DOWN_SAMPLE_KERNEL_SIZE = 3
DOWN_SAMPLE_STRIDE = 2
DOWN_SAMPLE_PADDING = 1
UP_SAMPLE_KERNEL_SIZE = 3
UP_SAMPLE_PADDING = 1

# Encoder and decoder block parameters
NUM_RES_BLOCKS = 2

# Residual block parameters
RES_BLOCK_KERNEL_SIZE = 3
RES_BLOCK_NUM_GROUPS = 4
RES_BLOCK_DROPOUT = 0.0

# Cross-attention parameters
CROSS_ATTN_NUM_HEADS = 4
CROSS_ATTN_SCALE = 0.1 
COND_CONTEXT_DIM = 32 # must match COND_OUTPUT_DIM in preprocess_config

# FiLM parameters (macro conditioning)
FILM_HIDDEN_DIM = 128
NUM_MACRO_SCALARS = 2

# U-Net model parameters
TIME_EMBED_DIM = 256
BASE_CHANNELS = 32
CHANNEL_MULT = [1, 2, 4, 8]
