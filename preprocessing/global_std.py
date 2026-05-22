"""
Compute the global scale factor used by preprocess_prices().
"""

import json
import torch
from pathlib import Path

from config import data_config, preprocess_config


def get_global_std(raw_data_dir: str = "../../data/raw") -> float:
    """
    Standard deviation of pooled log_returns * RETURN_SCALE_FACTOR across all training windows.
    """
    raw_dir = Path(raw_data_dir)
    all_windows = []

    for ticker in data_config.TICKERS:
        path = raw_dir / f"daily_data_{ticker}.json"

        # Load JSON file
        with open(path) as f:
            raw = json.load(f)

        # Extract time series
        time_series = raw["Time Series (Daily)"]
        dates = sorted(d for d in time_series if data_config.WINDOW_START <= d <= data_config.WINDOW_END)

        # Skip if not enough data
        if len(dates) < preprocess_config.T + 1:
            continue

        # Compute log returns
        closing = torch.tensor([float(time_series[d]["5. adjusted close"]) for d in dates])
        log_returns = torch.diff(torch.log(closing))
        for start in range(0, len(log_returns) - preprocess_config.T, preprocess_config.STRIDE):
            window = log_returns[start:start + preprocess_config.T] * preprocess_config.RETURN_SCALE_FACTOR
            all_windows.append(window)

    # Compute global std
    flat = torch.cat(all_windows)
    return float(flat.std().item())


def load_global_stats(path: str | Path = "../../data/train/global_stats.json") -> tuple[float, float]:
    """
    Load (global_std, scale_factor) written by the preprocess notebook.
    """
    with open(path) as f:
        d = json.load(f)
    sf = float(d.get("scale_factor", preprocess_config.RETURN_SCALE_FACTOR))

    return float(d["global_std"]), sf
