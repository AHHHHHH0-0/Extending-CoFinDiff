"""
Configuration for training.
"""

BATCH_SIZE = 32
LEARNING_RATE = 1e-5 # paper faithful
EPOCHS = 3000 # paper faithful
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 20
EARLY_STOPPING = 100 # paper faithful
P_UNCOND = 0.1  
