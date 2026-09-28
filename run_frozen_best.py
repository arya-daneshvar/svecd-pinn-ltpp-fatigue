"""
Run the frozen best pure PINN configuration.

This is the locked reference model selected after the Codex - First experiments:
  - dataset: fatigue_svecd_sdr_dense_state.csv
  - feature list: feature_cols_sdr_dense_state_obs30.txt
  - severity-state pure PINN
  - all-pairs trajectory loss
  - high-severity-aware band weights: 0.7,1.2,3.0

Examples:
  python3 run_frozen_best.py --parts temporal
  python3 run_frozen_best.py --parts all --suffix _my_comparison_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJ = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", choices=["all", "temporal", "screen"], default="all")
    parser.add_argument("--suffix", default="_frozen_high_only")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    env = os.environ.copy()
    env.update({
        "MPLCONFIGDIR": "/private/tmp",
        "SVECD_TAG": "_sdr_dense_state",
        "SVECD_FEATURE_TAG": "_sdr_dense_state_obs30",
        "STATE_PINN_OUT_SUFFIX": args.suffix,
        "STATE_PINN_RUN_PARTS": args.parts,
        "STATE_PINN_SEED_OFFSET": str(args.seed_offset),
        "STATE_PINN_W_PHYS_LIST": "0.1",
        "STATE_PINN_W_TRANS_LIST": "2",
        "STATE_PINN_W_TRAJ_LIST": "0.25",
        "STATE_PINN_TRAJ_PAIR_MODE": "all",
        "STATE_PINN_BAND_W": "0.7,1.2,3.0",
        "STATE_PINN_W_TOTAL": "0.4",
        "STATE_PINN_W_ONSET": "0.15",
        "STATE_PINN_W_CRACKED": "0.5",
        "STATE_PINN_HUBER_DELTA": "1.0",
        "STATE_PINN_SPLIT_LOCAL_DENSE": "1",
        "SVECD_ANCHOR": "1",
        "SVECD_HYBRID": "0",
        "SVECD_ENSEMBLE": "1",
        "OMP_NUM_THREADS": "1",
    })
    subprocess.run([sys.executable, "train_severity_state_pinn.py"], cwd=PROJ, env=env, check=True)


if __name__ == "__main__":
    main()
