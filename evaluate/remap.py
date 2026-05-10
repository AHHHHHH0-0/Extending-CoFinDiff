import numpy as np


def per_path_normalize(
    samples: np.ndarray,
    t_tgt: float,
    r_tgt: float,
    T: int,
) -> np.ndarray:
    """
    Remap each generated path so it exactly matches its target trend and
    realized volatility.

    The model outputs paths in a standardized (inverse-Haar) space.  A global
    affine remap (one std_w for all paths) systematically overshoots the target
    RV because the model's output variance is not exactly 1.  This function
    fixes that by normalizing each path individually:

      1. Center  – subtract the path's own mean.
      2. Scale   – multiply so the path's variance equals ``var_target``.
      3. Shift   – add the target mean ``mu``.

    After the transform every path satisfies:
      * sum(path * 100)        == t_tgt   (trend)
      * sum((path * 100) ** 2) == r_tgt   (realized volatility)

    Parameters
    ----------
    samples : np.ndarray, shape (N, T)
        Raw model output in standardized space.
    t_tgt : float
        Target cumulative return in percentage points (e.g. -20, 0, 20).
    r_tgt : float
        Target realized volatility (sum of squared percentage returns).
    T : int
        Number of time steps per path.

    Returns
    -------
    np.ndarray, shape (N, T)
        Calibrated paths in log-return scale.
    """
    mu = t_tgt / (100.0 * T)
    var_target = (r_tgt / (100.0 ** 2) - T * mu ** 2) / (T - 1)
    var_target = max(var_target, 0.0)

    centered = samples - samples.mean(axis=1, keepdims=True)
    path_var = (centered ** 2).sum(axis=1) / (T - 1)           # (N,)
    scale = np.sqrt(var_target / np.maximum(path_var, 1e-12))  # (N,)
    return centered * scale[:, None] + mu
