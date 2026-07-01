# Artifact Manifest

This repository is a frozen, paper-facing reproduction package rather than the
full exploratory thesis workspace.

## What Is Included

| path | role |
|---|---|
| `fatigue_svecd_sdr_dense_state.csv` | Frozen model-ready dataset, 2,879 rows plus header. |
| `fatigue_svecd_sdr_dense_state.parquet` | Same frozen dataset in Parquet format. |
| `feature_cols_sdr_dense_state_obs30.txt` | Frozen 41-feature model input list. |
| `frozen_config.json` | Machine-readable selected model configuration and reference metrics. |
| `run_frozen_best.py` | Stable runner for the selected configuration. |
| `train_severity_state_pinn.py` | Severity-state S-VECD PINN trainer. |
| `train_models.py` | Shared feature loading and utility functions used by the trainer. |
| `build_sdr_feature_dataset.py` | SDR feature-construction script. |
| `build_sdr_dense_feature_dataset.py` | Dense SDR feature-construction script. |
| `build_severity_state_dataset.py` | Disjoint low/moderate/high target-construction script. |
| `data_cache/ANALYSIS_DIS_AC.csv` | Raw LTPP distress table cached for auditing severity-state target construction. |
| `outputs/` | Curated reference outputs from the frozen run and robustness checks. |
| `reports/` | Short reports summarizing supporting ablations and selection decisions. |

## What Is Not Included

- The full local thesis workspace.
- The full raw SDR cache used during exploratory feature construction.
- Serialized neural-network checkpoints.
- Every exploratory experiment attempted before selecting the frozen model.

## Main Reproduction Target

The paper-facing model is the severity-aware pure S-VECD PINN with:

- dataset: `fatigue_svecd_sdr_dense_state.csv`
- features: `feature_cols_sdr_dense_state_obs30.txt`
- variant: `state_transition_w0.1_t2_traj0.25`
- loss weights: `w_phys=0.1`, `w_trans=2`, `w_traj=0.25`
- severity band weights: `0.7,1.2,3.0`
- target columns: `C_PCT_LOW_ONLY`, `C_PCT_MOD_ONLY`, `C_PCT_HIGH_ONLY`

The exact environment variables are stored in `frozen_config.json` and applied
by `run_frozen_best.py`.

## Reference Metrics

Single selected run:

| evaluation | total R2log |
|---|---:|
| temporal latest-survey holdout | 0.789 |
| cold unseen sections | 0.200 |
| first-anchor unseen sections | 0.517 |
| rolling-anchor unseen sections | 0.749 |

Five-seed stability mean:

| evaluation | total R2log mean |
|---|---:|
| temporal latest-survey holdout | 0.780 |
| cold unseen sections | 0.174 |
| first-anchor unseen sections | 0.521 |
| rolling-anchor unseen sections | 0.746 |

## Recommended Reader Workflow

1. Read `DATA_NOTICE.md` for data provenance and license boundaries.
2. Run `python3 check_artifact.py` to verify the frozen files.
3. Run `python3 run_frozen_best.py --parts temporal --suffix _reproduce_temporal`.
4. Compare the generated `temporal_state_sweep.csv` with the reference metrics
   in `outputs/outputs_severity_state_pinn_sdr_dense_state_obs30_traj_all_loss_high_only_full/`.
5. Run `python3 run_frozen_best.py --parts all --suffix _reproduce_full` for the
   full validation suite.

The trainer is stochastic. Small numerical differences are expected across
hardware, PyTorch versions, and random seeds.
