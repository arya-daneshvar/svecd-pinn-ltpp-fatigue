# Reproducibility

This document gives the shortest path from a fresh clone to regenerated metrics
for the paper-facing frozen artifact.

## Frozen Configuration

The frozen model configuration is stored in `frozen_config.json` and wrapped by `run_frozen_best.py`.

The selected model is:

```text
high_only_severity_aware_pure_pinn
dataset: fatigue_svecd_sdr_dense_state.csv
features: feature_cols_sdr_dense_state_obs30.txt
variant: state_transition_w0.1_t2_traj0.25
band weights: 0.7,1.2,3.0
```

The artifact contains 2,879 model-ready rows and 41 frozen input features.

## Artifact Check

Before installing modeling dependencies, verify that all frozen files are
present:

```bash
python3 check_artifact.py
```

## Commands

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional closer environment match:

```bash
pip install -r requirements-lock.txt
```

Run temporal validation:

```bash
python3 run_frozen_best.py --parts temporal --suffix _reproduce_temporal
```

Run full validation:

```bash
python3 run_frozen_best.py --parts all --suffix _reproduce_full
```

Run seed-stability check:

```bash
python3 run_robustness_seed_stability.py
```

Generated result folders are named
`outputs_severity_state_pinn_sdr_dense_state<suffix>/`. The curated reference
outputs committed to the repository are under `outputs/`.

## Expected Reference Metrics

The selected single-run model produced:

| evaluation | total R2log |
|---|---:|
| temporal | 0.789 |
| cold unseen | 0.200 |
| first anchor | 0.517 |
| rolling anchor | 0.749 |

The five-seed stability mean was:

| evaluation | total R2log mean |
|---|---:|
| temporal | 0.780 |
| cold unseen | 0.174 |
| first anchor | 0.521 |
| rolling anchor | 0.746 |

## Numerical Tolerance

The trainer reruns the model rather than loading a saved checkpoint. Small
differences can occur across hardware, PyTorch versions, BLAS backends, and
random seeds. For paper reproduction, compare rounded metrics and trends rather
than expecting bitwise equality.

## Data Rebuild Boundary

The final model-ready dataset is included. The scripts that produced the SDR
features and severity-state targets are also included, but the complete raw SDR
cache is not. The cached `data_cache/ANALYSIS_DIS_AC.csv` file is included so
the final target-construction step can be audited.
