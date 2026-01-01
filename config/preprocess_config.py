# Haar wavelet parameters
HAAR_WAVELET_LEVELS = 8 # assert log2(T) == levels
TARGET_SHAPE = (16, 16) # assert H * W == T

# Data preprocessing parameters
T = 256 # power of 2 for the Haar wavelet transform
RETURN_SCALE_FACTOR = 100.0

#TODO: 
# Condition encoder parameters
COND_TOKEN_DIM = 20  
COND_HIDDEN_DIM = 64 
NUM_CONDITION_TOKENS = 4  # trend, realized_vol, interest_rate, market volatility index
