# LTPP Severity-State Target Correction

The previous target treated fatigue severity bands mainly as cumulative modeling bands. That was useful mathematically, but it was not fully faithful to how LTPP defines alligator/fatigue cracking severity. LTPP stores low, moderate, and high fatigue cracking as disjoint visual severity states:

- `GATOR_CRACK_A_L`: low-severity area
- `GATOR_CRACK_A_M`: moderate-severity area
- `GATOR_CRACK_A_H`: high-severity area

The corrected dataset now trains on:

- `C_PCT_LOW_ONLY`
- `C_PCT_MOD_ONLY`
- `C_PCT_HIGH_ONLY`

The model still keeps the pure S-VECD PINN structure. Internally, the PINN produces ordered cumulative severity-or-higher states:

- `CUM_L = low + moderate + high`
- `CUM_M = moderate + high`
- `CUM_H = high`

Those are converted back into disjoint LTPP severity states for the data loss:

- `LOW_ONLY = CUM_L - CUM_M`
- `MOD_ONLY = CUM_M - CUM_H`
- `HIGH_ONLY = CUM_H`

This keeps the governing fatigue progression monotone in cumulative damage space while allowing low-only area to decline when distress progresses into moderate/high severity.

## Files Added

- `build_severity_state_dataset.py`
- `fatigue_svecd_sdr_dense_state.csv`
- `fatigue_svecd_sdr_dense_state.parquet`
- `feature_cols_sdr_dense_state.txt`
- `data_dictionary_sdr_dense_state.md`
- `outputs_severity_state_targets_sdr_dense_state.csv`
- `train_severity_state_pinn.py`
- `outputs_severity_state_pinn_sdr_dense_state/`

## Dataset

State-target rows from the raw LTPP distress fields: 2,872.

Model-ready rows after the usual PINN filters: 2,285 rows, 326 sections.

Target sparsity:

| target | zero share | nonzero median | max |
|---|---:|---:|---:|
| `C_PCT_LOW_ONLY` | 54.3% | 1.827 | 54.054 |
| `C_PCT_MOD_ONLY` | 78.2% | 2.314 | 83.156 |
| `C_PCT_HIGH_ONLY` | 89.6% | 1.878 | 86.490 |
| `C_PCT_TOTAL_STATE` | 49.0% | 3.299 | 86.845 |

## Best Pure PINN Setting

Best temporal sweep variant:

- `w_phys = 0.1`
- `w_trans = 2.0`

## Results

| scenario | metric | old cumulative PINN | corrected state PINN |
|---|---:|---:|---:|
| temporal | total `R2log` | 0.790 | 0.781 |
| temporal | cumulative moderate-or-higher `R2log` | 0.726 | 0.717 |
| temporal | high `R2log` | 0.514 | 0.460 |
| unseen sections, cold start | total `R2log` | 0.272 | 0.191 |
| unseen sections, first anchor | total `R2log` | 0.345 | 0.300 |
| unseen sections, rolling anchor | total `R2log` | 0.755 | 0.754 |

Corrected state-specific scores:

| scenario | low-only `R2log` | mod-only `R2log` | high-only `R2log` | mean severity class MAE |
|---|---:|---:|---:|---:|
| temporal | 0.598 | 0.571 | 0.460 | 0.301 |
| unseen sections, cold start | 0.097 | 0.082 | 0.004 | 0.467 |
| unseen sections, first anchor | 0.158 | 0.096 | -0.105 | 0.511 |
| unseen sections, rolling anchor | 0.544 | 0.490 | 0.373 | 0.343 |

## Interpretation

The corrected target fixes the conceptual problem: physics is no longer being asked to fit a target that blurs the LTPP severity definitions.

The important result is that rolling warm-start performance is almost unchanged in total cracking: `0.755` before vs `0.754` after target correction. That means the pure PINN can still extrapolate corrected fatigue progression when it has a recent section state.

The cold-start unseen-section result gets worse, which is expected and useful. The old target was smoother and easier. The corrected target exposes the real problem: without any section history, the model struggles to know where the section sits in the low/moderate/high severity progression.

High severity remains the hardest state because it is very sparse: almost 90% zeros, and when it appears it is usually a later-stage, section-specific transition.

The next pure-PINN refinement should be to feed the previous observed severity state into the anchor features, not only previous total cracking. The current transition correction already anchors all cumulative severity states mathematically, but the feature vector still only carries `LAST_OBS_CPCT_L`. Adding `LAST_OBS_CPCT_M` and `LAST_OBS_CPCT_H` would give the same pure PINN more faithful state information without becoming a hybrid or physics-guided residual model.
