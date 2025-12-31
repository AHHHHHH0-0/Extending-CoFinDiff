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

# U-Net model parameters
TIME_EMBED_DIM = 80
