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

The stored processed dataset contains 2,879 rows. The frozen trainer retains
2,285 model-ready rows across 326 sections after its required-field filters. The
frozen feature file contains 41 listed predictors, and three prior-condition
anchor variables are added at runtime.

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

Verify or regenerate the post-freeze validation package:

```bash
cd final_validation
python3 run_all_final_validations.py --resume
python3 plot_representative_frozen_pinn_trajectories.py
```

`final_validation_predictions.csv` contains the archived row-level
out-of-sample predictions used for uncertainty and reliability analyses.
`limited_data_predictions.csv` contains the fixed-subset limited-data
predictions. The two very large raw bootstrap-replicate matrices are not
included; their configurations, invalid-replicate log, compact summaries,
paired confidence intervals, and figures are archived.

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

Users who have a complete local SDR39 extraction can point the optional upstream
data-processing scripts to it with:

```bash
export LTPP_SDR_ROOT=/absolute/path/to/SDR
```

## Checkpoint Boundary

No serialized neural-network checkpoint was produced by the frozen experiment.
Reproduction therefore reruns the locked seed, data, code, preprocessing, and
configuration. Archived row-level predictions and evaluation tables permit
independent metric verification without retraining.
