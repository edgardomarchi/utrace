"""Run this when baselines need updating (e.g., after intentional algorithm change).
NEVER run silently as part of CI."""

from test_golden_mnist import _compute_golden_run as golden_run
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    results = golden_run()
    out_dir = Path(__file__).parent / 'baselines'
    out_dir.mkdir(exist_ok=True)
    for name, arr in results.items():
        np.save(out_dir / f'mean_{name}.npy', np.asarray(arr))
    print(f"Baselines saved to {out_dir}")