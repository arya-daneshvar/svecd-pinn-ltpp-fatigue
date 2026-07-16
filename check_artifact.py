#!/usr/bin/env python3
"""Check that the frozen reproduction artifact is complete.

This script is intentionally lightweight: it uses only the Python standard
library so readers can run it before installing the modeling dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "README.md",
    "ARTIFACT_MANIFEST.md",
    "FROZEN_MODEL_CARD.md",
    "REPRODUCIBILITY.md",
    "DATA_NOTICE.md",
    "frozen_config.json",
    "requirements.txt",
    "run_frozen_best.py",
    "train_severity_state_pinn.py",
    "train_models.py",
    "svecd.py",
    "build_sdr_feature_dataset.py",
    "build_sdr_dense_feature_dataset.py",
    "build_severity_state_dataset.py",
    "fatigue_svecd_sdr_dense_state.csv",
    "fatigue_svecd_sdr_dense_state.parquet",
    "feature_cols_sdr_dense_state_obs30.txt",
    "data_cache/ANALYSIS_DIS_AC.csv",
    "outputs/outputs_severity_state_pinn_sdr_dense_state_obs30_traj_all_loss_high_only_full/temporal_state_sweep.csv",
    "outputs/outputs_robustness_seed_stability/seed_stability_summary.csv",
    "final_validation/final_validation_predictions.csv",
    "final_validation/limited_data_predictions.csv",
    "final_validation/FINAL_VALIDATION_RESULTS.md",
    "final_validation/validation_utils.py",
    "final_validation/representative_frozen_pinn_trajectories.png",
    "reports/EQUATION_AUDIT.md",
]


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_first_data_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return next(reader)


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        print("Missing required artifact files:")
        for name in missing:
            print(f"  - {name}")
        return 1

    config = json.loads((ROOT / "frozen_config.json").read_text(encoding="utf-8"))
    dataset_rows = count_lines(ROOT / config["dataset"]) - 1
    feature_count = count_nonempty_lines(ROOT / config["feature_list"])
    temporal_row = read_first_data_row(
        ROOT / "outputs/outputs_severity_state_pinn_sdr_dense_state_obs30_traj_all_loss_high_only_full/temporal_state_sweep.csv"
    )

    print("Frozen artifact check passed.")
    print(f"Model: {config['frozen_name']}")
    print(f"Dataset: {config['dataset']} ({dataset_rows:,} rows)")
    print(f"Feature list: {config['feature_list']} ({feature_count} features)")
    print(f"Selected variant: {config['best_variant']}")
    print(f"Reference temporal R2log_total: {float(temporal_row['R2log_total']):.6f}")
    print(f"Reference temporal OnsetAUC: {float(temporal_row['OnsetAUC']):.6f}")
    prediction_rows = count_lines(ROOT / "final_validation/final_validation_predictions.csv") - 1
    print(f"Archived final-validation predictions: {prediction_rows:,} rows")
    print("Run `python3 run_frozen_best.py --parts temporal --suffix _reproduce_temporal` to regenerate the quick check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
