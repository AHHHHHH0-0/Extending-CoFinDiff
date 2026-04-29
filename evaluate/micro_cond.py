import numpy as np
from scipy.stats import pearsonr

def adherence_table(metric: str, model: str, data: dict, filenames: list) -> tuple:
    """Print a formatted target vs achieved table and return (targets, mean_achieved) lists."""
    rows = []
    # Collect data
    for fname in filenames:
        d = data[model][fname]
        target = d[f'{metric}_target']
        achieved = d[f'{metric}_achieved']
        rows.append((fname, target, achieved.mean(), achieved.std(), abs(achieved.mean() - target)))

    # Print table
    header = f"{model:20}  {'Target':>10}  {'Mean':>10}  {'Std':>10}  {'|Error|':>10}"
    print(header)
    print(f"{'─' * len(header)}")
    for fname, tgt, mean, std, err in rows:
        print(f"{fname:20}  {tgt:10.2f}  {mean:10.4f}  {std:10.4f}  {err:10.4f}")

    targets       = [r[1] for r in rows]
    mean_achieved = [r[2] for r in rows]
    mean_mae      = np.mean([r[4] for r in rows])
    r_val, _      = pearsonr(targets, mean_achieved)
    print(f"{'─' * len(header)}")
    print(f"Mean MAE: {mean_mae:<10.4f}    Pearson r: {r_val:<10.4f}")

    return targets, mean_achieved