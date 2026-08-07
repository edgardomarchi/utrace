"""Regenerate golden-test baselines.

Paths are resolved relative to THIS file (tests/integration/torch/), not the
current working directory, so the script works regardless of where it is run.

Usage:
    uv run --extra=cpu python tests/integration/torch/regenerate_baselines.py --api new

Run only for an intentional change to the algorithm or test setup. Commit the
regenerated .npy files alongside the change that motivated them, and explain
the reason in the commit message.
"""
import argparse
from pathlib import Path

import numpy as np

from _baselines import BASELINE_DIR

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', choices=['new'], default='new',
                        help="Which golden's baselines to regenerate.")
    args = parser.parse_args()

    from test_golden_mnist import _compute_golden_run
    results = _compute_golden_run()
    out_dir = BASELINE_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in results.items():
        np.save(out_dir / f'mean_{name}.npy', np.asarray(arr))
    print(f"Baselines for '{args.api}' API saved to {out_dir}")


if __name__ == '__main__':
    main()