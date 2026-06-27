# Feature Pruning Results for Severity-State Pure PINN

The corrected severity-state dataset has 261 model features in the full dense SDR feature list. Many of those are not actually observed often; they are dense because the builder filled missing SDR values from section, state-year, state, or global medians and then added `HAS_`/`TIER_` reliability flags.

This test keeps the dataset, target, and pure PINN formulation fixed, and swaps only the feature list through `SVECD_FEATURE_TAG`.

## Files Added or Changed

- `build_pruned_feature_sets.py`
- `run_pruned_feature_screen.py`
- `feature_pruned_sets.csv`
- `outputs_feature_pruning_state/temporal_feature_screen.csv`
- `feature_cols_sdr_dense_state_obs30.txt`
- `feature_cols_sdr_dense_state_obs20_flags.txt`
- `outputs_severity_state_pinn_sdr_dense_state_obs30_full/`
- `outputs_severity_state_pinn_sdr_dense_state_obs20_flags_full/`

`train_models.py` now supports `SVECD_FEATURE_TAG`, so one dataset tag can be paired with different feature-list tags.

## Feature Sets Screened

| feature set | features | idea |
|---|---:|---|
| full dense SDR | 261 | all dense SDR values plus all `HAS_`/`TIER_` flags |
| core20 | 20 | original mechanistic/material/traffic basics only |
| core24 | 24 | core20 plus binder/material extras |
| obs30 | 41 | core24 plus SDR values with at least 30% strict observed coverage |
| obs20 | 66 | core24 plus SDR values with at least 20% strict observed coverage |
| obs10 | 83 | core24 plus SDR values with at least 10% strict observed coverage |
| no_flags | 103 | all SDR value columns, but no `HAS_`/`TIER_` flags |
| obs20_flags | 150 | obs20 plus its `HAS_`/`TIER_` flags |
| climate_gpr | 60 | core24 plus climate and GPR blocks |
| climate_trf | 62 | core24 plus climate and monthly traffic blocks |

## Temporal Screen

All screen runs used fixed `w_phys=0.1`, `w_trans=2.0`.

| feature set | features | total `R2log` | onset AUC | low-only `R2log` | mod-only `R2log` | high-only `R2log` |
|---|---:|---:|---:|---:|---:|---:|
| obs30 | 41 | 0.792 | 0.885 | 0.606 | 0.566 | 0.468 |
| obs20_flags | 150 | 0.786 | 0.883 | 0.624 | 0.580 | 0.454 |
| full dense SDR | 261 | 0.781 | 0.874 | 0.598 | 0.571 | 0.460 |
| core20 | 20 | 0.771 | 0.869 | 0.610 | 0.560 | 0.461 |
| obs10 | 83 | 0.769 | 0.862 | 0.589 | 0.563 | 0.463 |
| obs20 | 66 | 0.769 | 0.860 | 0.594 | 0.559 | 0.446 |
| climate_trf | 62 | 0.768 | 0.863 | 0.599 | 0.590 | 0.444 |
| core24 | 24 | 0.768 | 0.868 | 0.604 | 0.557 | 0.432 |
| no_flags | 103 | 0.755 | 0.864 | 0.574 | 0.565 | 0.459 |
| climate_gpr | 60 | 0.750 | 0.844 | 0.580 | 0.562 | 0.451 |

## Full Validation of Top Candidates

| feature set | features | temporal total `R2log` | cold unseen total `R2log` | first-anchor total `R2log` | rolling-anchor total `R2log` |
|---|---:|---:|---:|---:|---:|
| full dense SDR | 261 | 0.781 | 0.191 | 0.300 | 0.754 |
| obs30 | 41 | 0.792 | 0.155 | 0.360 | 0.753 |
| obs20_flags | 150 | 0.786 | 0.201 | 0.331 | 0.747 |

Severity-specific full-validation scores:

| feature set | scenario | low-only `R2log` | mod-only `R2log` | high-only `R2log` |
|---|---|---:|---:|---:|
| full dense SDR | temporal | 0.598 | 0.571 | 0.460 |
| obs30 | temporal | 0.606 | 0.566 | 0.468 |
| obs20_flags | temporal | 0.624 | 0.580 | 0.454 |
| full dense SDR | rolling anchor | 0.544 | 0.490 | 0.373 |
| obs30 | rolling anchor | 0.546 | 0.477 | 0.352 |
| obs20_flags | rolling anchor | 0.538 | 0.466 | 0.379 |

## Recommended Default

Use `feature_cols_sdr_dense_state_obs30.txt` as the next default feature set for the severity-state PINN.

Why:

- It reduces the feature count from 261 to 41.
- It gives the best temporal total cracking score: `0.792` vs `0.781`.
- It gives the best temporal onset AUC: `0.885` vs `0.874`.
- It improves first-anchor unseen-section performance: `0.360` vs `0.300`.
- It preserves rolling-anchor performance: `0.753` vs `0.754`.

What it keeps:

- Core traffic, strain, modulus, layer thickness, and mixture features.
- Binder/material extras: `LOG_VISC_140`, `PEN_77`, `BINDER_GRADE_RANK`, `RELAX_SLOPE_M`.
- Drainage present flag.
- GPR AC thickness and dielectric features.
- Previous-year climate temperature and precipitation features.

What it drops:

- All `HAS_`/`TIER_` reliability flags.
- SMP pavement temperature/moisture/freeze features.
- GVW bin features.
- DLR strain-response features.
- Drainage pipe geometry and most drainage numeric fields.
- Cumulative climate fields.
- Monthly traffic count fields.

## Interpretation

Pruning helps, but only when it is strict. The useful cutoff here is roughly 30% strict observed SDR coverage. Lowering the cutoff to 20% or 10% adds many borrowed values and reduces temporal performance.

The `obs20_flags` set is worth keeping as a sensitivity model. It is not as compact, but it slightly improves cold-start unseen-section total `R2log` and gives the best temporal low/moderate disjoint severity scores. Still, its rolling-anchor total performance is below `obs30`, and it uses 150 features.

The cold-start unseen-section problem is not solved by feature pruning. That remains mostly a missing section-state problem: without any prior distress observation for the section, the model has weak information about where the section is in the low-to-moderate-to-high severity pathway.
