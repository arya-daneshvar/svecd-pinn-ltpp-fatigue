# Reproducibility

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

## Commands

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

