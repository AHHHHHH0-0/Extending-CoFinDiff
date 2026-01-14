# Haar wavelet parameters
HAAR_WAVELET_LEVELS = 8 # assert log2(T) == levels
TARGET_SHAPE = (16, 16) # assert H * W == T

# Data preprocessing parameters
T = 256 # power of 2 for the Haar wavelet transform
RETURN_SCALE_FACTOR = 100.0

# Condition encoder parameters
NUM_CONDITION_SCALARS = 4
COND_FC_HIDDEN_DIM = 256
COND_1D_CHANNELS = 64
COND_2D_CHANNELS = 32
COND_OUTPUT_DIM = 20 # must match COND_CONTEXT_DIM in denoiser_config
