"""
Run seed-stability robustness checks for the two current pure PINN candidates.

Candidates:
  - baseline trajectory loss: severity band weights 1.0,1.2,1.5
  - high_only severity-aware loss: severity band weights 0.7,1.2,3.0

The model family is otherwise fixed to the best all-pairs trajectory setup:
  - dataset: _sdr_dense_state
  - feature list: _sdr_dense_state_obs30
  - w_phys=0.10, w_trans=2, w_traj=0.25
  - trajectory mode: all
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJ = Path(__file__).resolve().parent
OUT = PROJ / "outputs_robustness_seed_stability"
OUT.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
MODELS = [
    {
        "name": "baseline",
        "band_w": "1.0,1.2,1.5",
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
]


def output_dir(model: str, seed: int) -> Path:
    suffix = f"_obs30_traj_all_robust_{model}_seed{seed}"
    return PROJ / f"outputs_severity_state_pinn_sdr_dense_state{suffix}"


def run_one(model: dict[str, str], seed: int) -> None:
    out_dir = output_dir(model["name"], seed)
    needed = [
        out_dir / "temporal_state_sweep.csv",
        out_dir / "scenario_a_state_unseen_sections.csv",
        out_dir / "warm_start_state_unseen_sections.csv",
    ]
    if all(p.exists() for p in needed):
        print(f"skip existing {model['name']} seed={seed}", flush=True)
        return

    suffix = f"_obs30_traj_all_robust_{model['name']}_seed{seed}"
    env = os.environ.copy()
    env.update({
        "MPLCONFIGDIR": "/private/tmp",
        "SVECD_TAG": "_sdr_dense_state",
        "SVECD_FEATURE_TAG": "_sdr_dense_state_obs30",
        "STATE_PINN_OUT_SUFFIX": suffix,
        "STATE_PINN_SEED_OFFSET": str(seed),
        "STATE_PINN_W_PHYS_LIST": "0.1",
        "STATE_PINN_W_TRANS_LIST": "2",
        "STATE_PINN_W_TRAJ_LIST": "0.25",
        "STATE_PINN_TRAJ_PAIR_MODE": "all",
        "STATE_PINN_BAND_W": model["band_w"],
        "STATE_PINN_W_TOTAL": model["w_total"],
        "STATE_PINN_W_ONSET": model["w_onset"],
        "STATE_PINN_W_CRACKED": model["w_cracked"],
        "STATE_PINN_HUBER_DELTA": model["huber"],
        "SVECD_ANCHOR": "1",
        "SVECD_HYBRID": "0",
        "SVECD_ENSEMBLE": "1",
        "OMP_NUM_THREADS": "1",
    })
    print(f"\n=== robustness seed run: {model['name']} seed={seed} ===", flush=True)
    subprocess.run([sys.executable, "train_severity_state_pinn.py"], cwd=PROJ, env=env, check=True)


def best_temporal(path: Path) -> dict:
    df = pd.read_csv(path / "temporal_state_sweep.csv")
    row = df[df["variant"].str.contains("transition_pred")].sort_values("R2log_total", ascending=False).iloc[0]
    return row.to_dict()


def collect() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            path = output_dir(model["name"], seed)
            t = best_temporal(path)
            t.update({"model": model["name"], "seed_offset": seed, "eval": "temporal"})
            rows.append(t)

            c = pd.read_csv(path / "scenario_a_state_unseen_sections.csv").iloc[0].to_dict()
            c.update({"model": model["name"], "seed_offset": seed, "eval": "cold_unseen"})
            rows.append(c)

            w = pd.read_csv(path / "warm_start_state_unseen_sections.csv")
            for label, pattern in [("first_anchor", "first_anchor"), ("rolling_anchor", "rolling_anchor")]:
                r = w[w["variant"].str.contains(pattern)].iloc[0].to_dict()
                r.update({"model": model["name"], "seed_offset": seed, "eval": label})
                rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "seed_stability_all_metrics.csv", index=False)
    return res


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "R2log_total",
        "OnsetAUC",
        "R2log_LOW_ONLY",
        "R2log_MOD_ONLY",
        "R2log_HIGH_ONLY",
        "R2log_CUM_M",
        "MAE_mean_class_cracked",
        "RMSE_total",
        "MAE_total",
    ]
    summary = (
        res.groupby(["model", "eval"])[metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(x) for x in col if str(x))
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(OUT / "seed_stability_summary.csv", index=False)
    return summary


def main() -> None:
    for seed in SEEDS:
        for model in MODELS:
            run_one(model, seed)
            partial = collect() if all(
                all((output_dir(m["name"], s) / fname).exists() for fname in [
                    "temporal_state_sweep.csv",
                    "scenario_a_state_unseen_sections.csv",
                    "warm_start_state_unseen_sections.csv",
                ])
                for s in SEEDS
                for m in MODELS
            ) else None
            if partial is not None:
                summarize(partial)
    res = collect()
    summary = summarize(res)
    key = summary[summary["eval"].isin(["temporal", "cold_unseen", "first_anchor", "rolling_anchor"])]
    cols = [
        "model",
        "eval",
        "R2log_total_mean",
        "R2log_total_std",
        "R2log_HIGH_ONLY_mean",
        "R2log_HIGH_ONLY_std",
        "OnsetAUC_mean",
        "OnsetAUC_std",
    ]
    print("\n=== seed-stability summary ===")
    print(key[cols].to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

