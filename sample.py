"""
Sampling script for CoFinDiff.

Generate financial time series samples using a trained model.
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from cofindiff import CoFinDiffTCN, Diffusion
from cofindiff.diffusion import generate_samples


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    macro_encoder: torch.nn.Module,
    device: str
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    macro_encoder.load_state_dict(checkpoint["macro_encoder_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model, macro_encoder


def main():
    parser = argparse.ArgumentParser(description="Sample from CoFinDiff")
    
    # Model arguments (must match training)
    parser.add_argument("--num_assets", type=int, default=5)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--macro_dim", type=int, default=10)
    
    # Diffusion arguments
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    
    # Sampling arguments
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="samples.npy")
    
    # Conditioning
    parser.add_argument("--macro_file", type=str, default=None,
                        help="Path to numpy file with macro conditions")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    device = args.device
    
    # Create model
    print("Creating model...")
    model = CoFinDiffTCN(
        in_channels=args.num_assets,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        time_dim=args.time_dim,
        cond_dim=args.cond_dim
    ).to(device)
    
    macro_encoder = MacroEncoder(
        input_dim=args.macro_dim,
        hidden_dim=256,
        output_dim=args.cond_dim
    ).to(device)
    
    # Load checkpoint
    model, macro_encoder = load_checkpoint(
        args.checkpoint, model, macro_encoder, device
    )
    
    model.eval()
    macro_encoder.eval()
    
    # Create diffusion
    diffusion = Diffusion(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        device=device
    )
    
    # Load or generate macro conditions
    if args.macro_file is not None:
        print(f"Loading macro conditions from {args.macro_file}")
        macro = torch.from_numpy(np.load(args.macro_file)).float().to(device)
    else:
        print("Using random macro conditions")
        macro = torch.randn(args.num_samples, args.macro_dim, device=device)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples with guidance scale {args.guidance_scale}...")
    
    with torch.no_grad():
        samples = generate_samples(
            model=model,
            diffusion=diffusion,
            macro=macro,
            macro_encoder=macro_encoder,
            seq_length=args.seq_length,
            num_channels=args.num_assets,
            guidance_scale=args.guidance_scale
        )
    
    # Save samples
    samples_np = samples.cpu().numpy()
    np.save(args.output, samples_np)
    
    print(f"Saved {samples_np.shape[0]} samples to {args.output}")
    print(f"Shape: {samples_np.shape}")
    print(f"Statistics: mean={samples_np.mean():.6f}, std={samples_np.std():.6f}")
    print(f"           min={samples_np.min():.6f}, max={samples_np.max():.6f}")


if __name__ == "__main__":
    main()

