"""
Screen severity-aware loss settings for the corrected severity-state pure PINN.

The model family is fixed to the best early-anchor trajectory setup:
  - dataset: _sdr_dense_state
  - feature list: _sdr_dense_state_obs30
  - w_phys=0.10, w_trans=2, w_traj=0.25
  - trajectory mode: all

Only the loss-shaping parameters vary.

Run:
  python3 run_severity_loss_sweep.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJ = Path(__file__).resolve().parent
OUT = PROJ / "outputs_severity_loss_sweep"
OUT.mkdir(exist_ok=True)


CONFIGS = [
    {
        "name": "baseline",
        "band_w": "1.0,1.2,1.5",
        "w_total": "0.4",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "1.0",
    },
    {
        "name": "high_mild",
        "band_w": "1.0,1.5,2.0",
        "w_total": "0.4",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "1.0",
    },
    {
        "name": "high_strong",
        "band_w": "1.0,2.0,3.0",
        "w_total": "0.4",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "1.0",
    },
    {
        "name": "mod_high_strong",
        "band_w": "0.8,1.8,3.0",
        "w_total": "0.4",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "1.0",
    },
    {
        "name": "high_only",
        "band_w": "0.7,1.2,3.0",
        "w_total": "0.4",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "1.0",
    },
    {
        "name": "state_heavy",
        "band_w": "1.0,1.8,2.5",
        "w_total": "0.2",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "1.0",
    },
    {
        "name": "total_heavy",
        "band_w": "1.0,1.5,2.5",
        "w_total": "0.7",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "1.0",
    },
    {
        "name": "cracked_heavy",
        "band_w": "1.0,1.5,2.5",
        "w_total": "0.4",
        "w_onset": "0.15",
        "w_cracked": "1.0",
        "huber": "1.0",
    },
    {
        "name": "onset_heavy",
        "band_w": "1.0,1.5,2.5",
        "w_total": "0.4",
        "w_onset": "0.30",
        "w_cracked": "0.5",
        "huber": "1.0",
    },
    {
        "name": "huber_small",
        "band_w": "1.0,1.5,2.5",
        "w_total": "0.4",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "0.5",
    },
    {
        "name": "huber_large",
        "band_w": "1.0,1.5,2.5",
        "w_total": "0.4",
        "w_onset": "0.15",
        "w_cracked": "0.5",
        "huber": "2.0",
    },
]


def run_one(cfg: dict[str, str]) -> dict:
    suffix = f"_obs30_traj_all_loss_{cfg['name']}"
    env = os.environ.copy()
    env.update({
        "MPLCONFIGDIR": "/private/tmp",
        "SVECD_TAG": "_sdr_dense_state",
        "SVECD_FEATURE_TAG": "_sdr_dense_state_obs30",
        "STATE_PINN_OUT_SUFFIX": suffix,
        "STATE_PINN_RUN_PARTS": "temporal",
        "STATE_PINN_W_PHYS_LIST": "0.1",
        "STATE_PINN_W_TRANS_LIST": "2",
        "STATE_PINN_W_TRAJ_LIST": "0.25",
        "STATE_PINN_TRAJ_PAIR_MODE": "all",
        "STATE_PINN_BAND_W": cfg["band_w"],
        "STATE_PINN_W_TOTAL": cfg["w_total"],
        "STATE_PINN_W_ONSET": cfg["w_onset"],
        "STATE_PINN_W_CRACKED": cfg["w_cracked"],
        "STATE_PINN_HUBER_DELTA": cfg["huber"],
        "SVECD_ANCHOR": "1",
        "SVECD_HYBRID": "0",
        "SVECD_ENSEMBLE": "1",
        "OMP_NUM_THREADS": "1",
    })
    print(f"\n=== severity loss screen: {cfg['name']} ===", flush=True)
    subprocess.run([sys.executable, "train_severity_state_pinn.py"], cwd=PROJ, env=env, check=True)

    out_dir = PROJ / f"outputs_severity_state_pinn_sdr_dense_state{suffix}"
    best = pd.read_csv(out_dir / "temporal_state_sweep.csv").sort_values("R2log_total", ascending=False).iloc[0].to_dict()
    best.update({
        "loss_config": cfg["name"],
        "band_w_cfg": cfg["band_w"],
        "w_total_cfg": float(cfg["w_total"]),
        "w_onset_cfg": float(cfg["w_onset"]),
        "w_cracked_cfg": float(cfg["w_cracked"]),
        "huber_cfg": float(cfg["huber"]),
        "output_dir": str(out_dir.relative_to(PROJ)),
    })
    return best


def main() -> None:
    rows = []
    for cfg in CONFIGS:
        rows.append(run_one(cfg))
        pd.DataFrame(rows).to_csv(OUT / "temporal_loss_sweep_partial.csv", index=False)
    res = pd.DataFrame(rows)
    res["severity_mean"] = res[["R2log_LOW_ONLY", "R2log_MOD_ONLY", "R2log_HIGH_ONLY"]].mean(axis=1)
    res["score_balanced"] = (
        0.40 * res["R2log_total"]
        + 0.20 * res["R2log_MOD_ONLY"]
        + 0.25 * res["R2log_HIGH_ONLY"]
        + 0.15 * res["OnsetAUC"]
    )
    res = res.sort_values("score_balanced", ascending=False).reset_index(drop=True)
    res.to_csv(OUT / "temporal_loss_sweep.csv", index=False)
    cols = [
        "loss_config",
        "band_w_cfg",
        "w_total_cfg",
        "w_onset_cfg",
        "w_cracked_cfg",
        "huber_cfg",
        "R2log_total",
        "OnsetAUC",
        "R2log_LOW_ONLY",
        "R2log_MOD_ONLY",
        "R2log_HIGH_ONLY",
        "severity_mean",
        "score_balanced",
    ]
    print("\n=== severity loss temporal screen ===")
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
