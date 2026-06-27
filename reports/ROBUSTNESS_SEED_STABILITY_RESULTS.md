# Robustness: Seed Stability

## Purpose

This is the first robustness check for the current pure PINN setup. The question here is not whether the model handles noisy data yet; it is whether the apparent advantage of the severity-aware loss survives different random initializations.

Compared models:

- `baseline`: all-pairs trajectory PINN with severity band weights `1.0 / 1.2 / 1.5`
- `high_only`: all-pairs trajectory PINN with severity band weights `0.7 / 1.2 / 3.0`

Fixed settings:

- Dataset: `fatigue_svecd_sdr_dense_state.csv`
- Feature set: `feature_cols_sdr_dense_state_obs30.txt`
- `w_phys=0.10`
- `w_trans=2`
- `w_traj=0.25`
- trajectory mode: `all`
- seeds: `0, 1, 2, 3, 4`

Outputs:

- All per-seed metrics: `outputs_robustness_seed_stability/seed_stability_all_metrics.csv`
- Mean/std/min/max summary: `outputs_robustness_seed_stability/seed_stability_summary.csv`
- Runner: `run_robustness_seed_stability.py`

## Total-Cracking R2log Stability

| model | eval | mean | std | min | max |
|---|---|---:|---:|---:|---:|
| baseline | temporal | 0.779 | 0.014 | 0.760 | 0.798 |
| high_only | temporal | 0.780 | 0.007 | 0.773 | 0.789 |
| baseline | cold unseen | 0.181 | 0.017 | 0.155 | 0.197 |
| high_only | cold unseen | 0.174 | 0.028 | 0.128 | 0.200 |
| baseline | first anchor | 0.509 | 0.027 | 0.478 | 0.544 |
| high_only | first anchor | 0.521 | 0.023 | 0.488 | 0.552 |
| baseline | rolling anchor | 0.742 | 0.006 | 0.735 | 0.748 |
| high_only | rolling anchor | 0.746 | 0.006 | 0.736 | 0.750 |

Mean total-cracking differences, `high_only - baseline`:

- temporal: `+0.001`
- cold unseen: `-0.006`
- first anchor: `+0.012`
- rolling anchor: `+0.004`

## High-Severity Stability

| model | eval | high-only R2log mean | high-only R2log std |
|---|---|---:|---:|
| baseline | temporal | 0.459 | 0.023 |
| high_only | temporal | 0.490 | 0.023 |
| baseline | cold unseen | -0.010 | 0.025 |
| high_only | cold unseen | -0.013 | 0.056 |
| baseline | first anchor | -0.107 | 0.069 |
| high_only | first anchor | -0.020 | 0.058 |
| baseline | rolling anchor | 0.353 | 0.011 |
| high_only | rolling anchor | 0.373 | 0.008 |

Mean high-severity differences, `high_only - baseline`:

- temporal: `+0.032`
- cold unseen: `-0.003`
- first anchor: `+0.087`
- rolling anchor: `+0.020`

## Onset AUC Stability

| model | eval | onset AUC mean | onset AUC std |
|---|---|---:|---:|
| baseline | temporal | 0.871 | 0.026 |
| high_only | temporal | 0.868 | 0.012 |
| baseline | cold unseen | 0.752 | 0.006 |
| high_only | cold unseen | 0.750 | 0.007 |
| baseline | first anchor | 0.814 | 0.007 |
| high_only | first anchor | 0.817 | 0.008 |
| baseline | rolling anchor | 0.846 | 0.005 |
| high_only | rolling anchor | 0.849 | 0.006 |

## Interpretation

The `high_only` loss does not simply win because of one lucky seed. Across five seeds:

- It matches the baseline on temporal total-cracking performance.
- It is slightly worse on cold unseen total `R2log`, but the difference is small relative to the cold-split variance.
- It is consistently better for warm-start forecasting, especially first-anchor prediction.
- It is meaningfully better for high-severity prediction when at least one observed anchor is available.
- It has lower temporal variance than the baseline, suggesting the severity-aware loss may be a more stable training objective.

The weak spot remains cold-start unseen-section generalization. Neither loss solves that. The model still needs either better section descriptors, better segmentation, or a cold-start-specific training/evaluation strategy.

## Recommendation

Keep `high_only` as the recommended practical forecasting model for anchored use cases.

Use the baseline trajectory model as the temporal reference model.

For the next robustness layer, run controlled degradation tests on both models:

- label noise on severity states: `5%, 10%, 20%`
- reduced training sections: `75%, 50%, 25%`
- anchor quality degradation: first anchor only, noisy anchor, delayed anchor
- feature missingness/dropout on sparse SDR groups: climate, GPR, traffic/material subsets

