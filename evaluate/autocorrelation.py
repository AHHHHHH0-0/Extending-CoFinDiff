"""
Autocorrelation function for squared returns.
"""

import numpy as np


def acf_squared(X: np.ndarray, lags) -> np.ndarray:
    """
    Mean autocorrelation of squared returns across sequences at specified lags.

    Formula (applied to r²):
        ρ(τ) = Σ_{t=1}^{T-τ} (r²_t − r̄²)(r²_{t+τ} − r̄²)
               ─────────────────────────────────────────────
               Σ_{t=1}^{T} (r²_t − r̄²)²
    """
    r2 = X ** 2
    mu_r2 = r2.mean(axis=1, keepdims=True)
    centered = r2 - mu_r2
    denom = (centered ** 2).sum(axis=1)   # (N,)

    results = []
    for tau in lags:
        numer = (centered[:, :-tau] * centered[:, tau:]).sum(axis=1)
        results.append((numer / denom).mean())
    return np.array(results)
