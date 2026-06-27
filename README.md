# Severity-Aware S-VECD PINN for LTPP Fatigue Cracking

This repository freezes the best pure PINN model from the thesis fatigue-cracking experiments for reproducible comparison against other modeling streamlines and data-degradation scenarios.

The frozen model is a severity-aware S-VECD physics-informed neural network trained on LTPP-derived fatigue cracking severity states.

## Frozen Model

- Model: pure S-VECD PINN
- Dataset: `fatigue_svecd_sdr_dense_state.csv`
- Feature list: `feature_cols_sdr_dense_state_obs30.txt`
- Targets:
  - `C_PCT_LOW_ONLY`
  - `C_PCT_MOD_ONLY`
  - `C_PCT_HIGH_ONLY`
- Internal cumulative states:
  - `C_PCT_L = low + moderate + high`
  - `C_PCT_M = moderate + high`
  - `C_PCT_H = high`
- Selected variant: `state_transition_w0.1_t2_traj0.25`
- Severity band weights: `0.7 / 1.2 / 3.0`
- Trajectory loss: all prior/future pairs, `w_traj=0.25`
- Physics weight: `w_phys=0.10`
- Transition weight: `w_trans=2`

See [FROZEN_MODEL_CARD.md](FROZEN_MODEL_CARD.md) and [frozen_config.json](frozen_config.json) for the locked configuration.

## Results Snapshot

Single full-validation run:

| model | temporal R2log | cold unseen R2log | first-anchor R2log | rolling-anchor R2log |
|---|---:|---:|---:|---:|
| frozen severity-aware PINN | 0.789 | 0.200 | 0.517 | 0.749 |

Seed-stability mean across five seeds:

| model | temporal R2log | cold unseen R2log | first-anchor R2log | rolling-anchor R2log |
|---|---:|---:|---:|---:|
| frozen severity-aware PINN | 0.780 | 0.174 | 0.521 | 0.746 |

Detailed outputs are in `outputs/`, and supporting experiment reports are in `reports/`.

## Repository Layout

```text
.
├── README.md
├── FROZEN_MODEL_CARD.md
├── frozen_config.json
├── run_frozen_best.py
├── train_severity_state_pinn.py
├── train_models.py
├── build_severity_state_dataset.py
├── build_sdr_dense_feature_dataset.py
├── build_sdr_feature_dataset.py
├── fatigue_svecd_sdr_dense_state.csv
├── feature_cols_sdr_dense_state_obs30.txt
├── data_cache/ANALYSIS_DIS_AC.csv
├── outputs/
└── reports/
```

The training scripts expect the frozen dataset and feature-list files in the repository root. That is why the primary dataset files are stored at the top level.

## Install

Python 3.11 was used during the frozen experiments. A recent Python 3.10+ environment should also work.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

Temporal-only quick check:

```bash
python3 run_frozen_best.py --parts temporal --suffix _frozen_temporal_check
```

Full validation:

```bash
python3 run_frozen_best.py --parts all --suffix _frozen_full
```

The trainer writes results to `outputs_severity_state_pinn_sdr_dense_state<suffix>/`.

## Data Provenance

The processed dataset is derived from the FHWA Long-Term Pavement Performance program / LTPP InfoPave Standard Data Release data. The file `data_cache/ANALYSIS_DIS_AC.csv` is included so that the final disjoint severity-state target construction can be audited.

See [DATA_NOTICE.md](DATA_NOTICE.md) for attribution and redistribution notes.

## Citation

If you use this repository, cite both:

1. the original FHWA LTPP / InfoPave data source, and
2. the archived release of this repository.

Before publication, create a GitHub release and archive it with Zenodo or OSF to obtain a DOI. Update [CITATION.cff](CITATION.cff) with the final author metadata and DOI.

## Note On Checkpoints

This repository freezes the dataset, code, configuration, and validation outputs. It does not include serialized trained weights because the trainer was designed to report reproducible evaluation runs rather than save checkpoints.

