"""
Configuration for the diffusion process.
"""

# Diffusion parameters
TIMESTEPS = 1000
OBJECTIVE = 'pred_noise'
BETA_SCHEDULE = 'linear' # wandb sweep
BETA_START = 1e-4
BETA_END = 0.02
AUTO_NORMALIZE = False

# Sampling parameters
GUIDANCE_SCALE = 1.0 
RETURN_TRAJECTORY = False
