# Frozen PINN Model Card

## Frozen Reference

This repository freezes the selected pure PINN from the thesis experiments for
paper reproduction and downstream comparisons against other modeling
streamlines and data-degradation experiments.

The frozen model is:

- **Name:** `high_only_severity_aware_pure_pinn`
- **Dataset:** `fatigue_svecd_sdr_dense_state.csv`
- **Feature list:** `feature_cols_sdr_dense_state_obs30.txt`
- **Target:** disjoint LTPP fatigue severity states
  - `C_PCT_LOW_ONLY`
  - `C_PCT_MOD_ONLY`
  - `C_PCT_HIGH_ONLY`
- **Internal state:** ordered cumulative severity-or-greater states
  - `C_PCT_L = low + moderate + high`
  - `C_PCT_M = moderate + high`
  - `C_PCT_H = high`
- **Model class:** pure S-VECD PINN
- **Best variant:** `state_transition_w0.1_t2_traj0.25`

This is not a partial PINN, physics-guided model, or hybrid external learner. The learned function is the neural S-VECD PINN trained directly against severity-state targets, with physics/monotonicity/transition losses inside the PINN objective.

## Locked Configuration

```text
SVECD_TAG=_sdr_dense_state
SVECD_FEATURE_TAG=_sdr_dense_state_obs30
SVECD_ANCHOR=1
SVECD_HYBRID=0
SVECD_ENSEMBLE=1

STATE_PINN_W_PHYS_LIST=0.1
STATE_PINN_W_TRANS_LIST=2
STATE_PINN_W_TRAJ_LIST=0.25
STATE_PINN_TRAJ_PAIR_MODE=all
STATE_PINN_BAND_W=0.7,1.2,3.0
STATE_PINN_W_TOTAL=0.4
STATE_PINN_W_ONSET=0.15
STATE_PINN_W_CRACKED=0.5
STATE_PINN_HUBER_DELTA=1.0
```

## Why This Model Was Frozen

The temporal-only baseline still has slightly stronger single-run temporal fit, but the `high_only` severity-aware loss is the better practical forecasting reference:

| model | temporal R2log | cold unseen R2log | first-anchor R2log | rolling-anchor R2log |
|---|---:|---:|---:|---:|
| baseline trajectory | 0.798 | 0.155 | 0.497 | 0.735 |
| high_only | 0.789 | 0.200 | 0.517 | 0.749 |

Seed-stability means across five seeds:

| model | temporal R2log | cold unseen R2log | first-anchor R2log | rolling-anchor R2log |
|---|---:|---:|---:|---:|
| baseline | 0.779 | 0.181 | 0.509 | 0.742 |
| high_only | 0.780 | 0.174 | 0.521 | 0.746 |

High-severity anchored behavior is the main reason to prefer `high_only`:

| model | first-anchor high-only R2log | rolling-anchor high-only R2log |
|---|---:|---:|
| baseline | -0.107 | 0.353 |
| high_only | -0.020 | 0.373 |

## Folder Contents

- `ARTIFACT_MANIFEST.md`: inventory of included and excluded reproducibility files.
- `check_artifact.py`: dependency-free completeness check for the frozen artifact.
- `run_frozen_best.py`: stable runner for the frozen model.
- `train_severity_state_pinn.py`: frozen trainer.
- `train_models.py`: shared PINN/data utilities used by the trainer.
- `build_severity_state_dataset.py`: target maker for LTPP disjoint severity states.
- `build_sdr_dense_feature_dataset.py`: dense SDR feature dataset maker.
- `build_sdr_feature_dataset.py`: SDR feature provenance script.
- `fatigue_svecd_sdr_dense_state.csv`: frozen model dataset.
- `feature_cols_sdr_dense_state_obs30.txt`: frozen model feature list.
- `data_cache/ANALYSIS_DIS_AC.csv`: cached raw distress table needed to rebuild severity-state targets.
- `outputs/`: selected validation, sweep, and robustness outputs.
- `reports/`: supporting reports from the exploratory phase.

## Reproduce The Frozen Model

Completeness check:

```bash
python3 check_artifact.py
```

Temporal-only quick check:

```bash
python3 run_frozen_best.py --parts temporal --suffix _frozen_temporal_check
```

Full validation:

```bash
python3 run_frozen_best.py --parts all --suffix _frozen_full
```

The trainer does not currently serialize weights. This frozen reference is
therefore a reproducible recipe and dataset freeze, not a saved checkpoint.

## Intended Use

Use this folder as the stable PINN reference when comparing against:

- non-PINN streamlines,
- ML baselines,
- survival/hazard formulations,
- limited-data scenarios,
- label-noise scenarios,
- feature-missingness scenarios.

The next experiments should run degradation conditions against this frozen PINN and the competing streamlines without changing the frozen model configuration.
