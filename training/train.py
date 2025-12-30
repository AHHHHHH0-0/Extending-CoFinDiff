"""
Training script for CoFinDiff.

This script demonstrates how to train the CoFinDiff model on financial
time series data with macroeconomic conditioning.
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from cofindiff import CoFinDiffTCN, Diffusion
from cofindiff.model import MacroEncoder
from cofindiff.diffusion import train_step, generate_samples


class SyntheticFinancialDataset(Dataset):
    """
    Synthetic financial dataset for testing.
    
    Generates synthetic financial time series with macro conditioning.
    Replace this with your actual financial data loader.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 64,
        num_assets: int = 5,
        macro_dim: int = 10,
        seed: int = 42
    ):
        super().__init__()
        
        np.random.seed(seed)
        
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_assets = num_assets
        self.macro_dim = macro_dim
        
        # Generate synthetic macro conditions
        self.macro = np.random.randn(num_samples, macro_dim).astype(np.float32)
        
        # Generate synthetic returns that depend on macro
        self.returns = self._generate_returns()
        
    def _generate_returns(self) -> np.ndarray:
        """Generate synthetic returns conditioned on macro."""
        returns = np.zeros(
            (self.num_samples, self.num_assets, self.seq_length),
            dtype=np.float32
        )
        
        for i in range(self.num_samples):
            # Macro influence on volatility
            vol = 0.01 * (1 + 0.5 * np.abs(self.macro[i, 0]))
            
            # Macro influence on trend
            trend = 0.0001 * self.macro[i, 1]
            
            # Generate correlated returns
            for j in range(self.num_assets):
                noise = np.random.randn(self.seq_length)
                returns[i, j] = trend + vol * noise
                
        return returns
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.returns[idx]),
            torch.from_numpy(self.macro[idx])
        )


def train(
    model: nn.Module,
    macro_encoder: nn.Module,
    diffusion: Diffusion,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: str,
    p_uncond: float = 0.1,
    save_dir: str = "checkpoints",
    log_interval: int = 100
):
    """
    Train the CoFinDiff model.
    
    Args:
        model: CoFinDiff denoiser
        macro_encoder: Macro encoder network
        diffusion: Diffusion object
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        device: Device to train on
        p_uncond: Probability of dropping conditioning
        save_dir: Directory to save checkpoints
        log_interval: How often to log loss
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    model.train()
    macro_encoder.train()
    
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (x0, macro) in enumerate(pbar):
            x0 = x0.to(device)
            macro = macro.to(device)
            
            loss = train_step(
                model=model,
                diffusion=diffusion,
                x0=x0,
                macro=macro,
                macro_encoder=macro_encoder,
                optimizer=optimizer,
                p_uncond=p_uncond
            )
            
            epoch_loss += loss
            num_batches += 1
            global_step += 1
            
            if global_step % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss:.6f}"})
                
        if scheduler is not None:
            scheduler.step()
            
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "macro_encoder_state_dict": macro_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            )


def main():
    parser = argparse.ArgumentParser(description="Train CoFinDiff")
    
    # Data arguments
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--num_assets", type=int, default=5)
    parser.add_argument("--macro_dim", type=int, default=10)
    
    # Model arguments
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=128)
    
    # Diffusion arguments
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--p_uncond", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    device = args.device
    
    # Create dataset
    print("Creating synthetic dataset...")
    dataset = SyntheticFinancialDataset(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        num_assets=args.num_assets,
        macro_dim=args.macro_dim
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
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
    
    # Create diffusion
    diffusion = Diffusion(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        device=device
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in macro_encoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    print(f"Model receptive field: {model.get_receptive_field()} timesteps")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(macro_encoder.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    
    # Train
    print("Starting training...")
    train(
        model=model,
        macro_encoder=macro_encoder,
        diffusion=diffusion,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        p_uncond=args.p_uncond,
        save_dir=args.save_dir
    )
    
    print("Training complete!")
    
    # Generate samples for verification
    print("\nGenerating sample outputs...")
    model.eval()
    macro_encoder.eval()
    
    with torch.no_grad():
        # Get a batch of macro conditions
        sample_macro = torch.randn(4, args.macro_dim, device=device)
        
        samples = generate_samples(
            model=model,
            diffusion=diffusion,
            macro=sample_macro,
            macro_encoder=macro_encoder,
            seq_length=args.seq_length,
            num_channels=args.num_assets,
            guidance_scale=3.0
        )
        
        print(f"Generated samples shape: {samples.shape}")
        print(f"Sample statistics: mean={samples.mean():.4f}, std={samples.std():.4f}")


if __name__ == "__main__":
    main()

