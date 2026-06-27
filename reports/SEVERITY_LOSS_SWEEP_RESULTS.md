# Severity-Aware Loss Sweep Results

## Purpose

We tested whether the corrected LTPP severity-state PINN benefits from making the data loss more severity-aware. The model is still the same pure PINN family: no partial PINN, no physics-guided external model, no hybrid target. The sweep only changes loss shaping.

Fixed model family:

- Dataset: `fatigue_svecd_sdr_dense_state.csv`
- Feature set: `feature_cols_sdr_dense_state_obs30.txt`
- Model: severity-state pure PINN with cumulative ordered internal states
- Anchoring: on
- Trajectory loss: all-pairs, `w_traj=0.25`
- Physics weight: `w_phys=0.10`
- Transition weight: `w_trans=2`

Outputs:

- Temporal sweep: `outputs_severity_loss_sweep/temporal_loss_sweep.csv`
- Full comparison: `outputs_severity_loss_sweep/full_validation_comparison.csv`
- Full high-only run: `outputs_severity_state_pinn_sdr_dense_state_obs30_traj_all_loss_high_only_full`
- Full high-mild run: `outputs_severity_state_pinn_sdr_dense_state_obs30_traj_all_loss_high_mild_full`

## Temporal Screen

Balanced score used for screening:

`0.40*R2log_total + 0.20*R2log_MOD_ONLY + 0.25*R2log_HIGH_ONLY + 0.15*OnsetAUC`

| config | band weights low/mod/high | total R2log | onset AUC | low-only R2log | mod-only R2log | high-only R2log | balanced score |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.0 / 1.2 / 1.5 | 0.798 | 0.898 | 0.577 | 0.567 | 0.491 | 0.690 |
| high_only | 0.7 / 1.2 / 3.0 | 0.789 | 0.866 | 0.571 | 0.556 | 0.505 | 0.683 |
| high_mild | 1.0 / 1.5 / 2.0 | 0.796 | 0.874 | 0.595 | 0.575 | 0.473 | 0.683 |
| onset_heavy | 1.0 / 1.5 / 2.5 | 0.796 | 0.872 | 0.601 | 0.569 | 0.476 | 0.682 |
| huber_large | 1.0 / 1.5 / 2.5 | 0.795 | 0.890 | 0.592 | 0.583 | 0.454 | 0.682 |
| mod_high_strong | 0.8 / 1.8 / 3.0 | 0.778 | 0.852 | 0.581 | 0.563 | 0.502 | 0.677 |
| high_strong | 1.0 / 2.0 / 3.0 | 0.783 | 0.869 | 0.606 | 0.576 | 0.471 | 0.677 |
| cracked_heavy | 1.0 / 1.5 / 2.5 | 0.796 | 0.864 | 0.601 | 0.579 | 0.442 | 0.675 |
| total_heavy | 1.0 / 1.5 / 2.5 | 0.783 | 0.862 | 0.600 | 0.577 | 0.449 | 0.670 |
| state_heavy | 1.0 / 1.8 / 2.5 | 0.774 | 0.861 | 0.605 | 0.566 | 0.466 | 0.668 |
| huber_small | 1.0 / 1.5 / 2.5 | 0.771 | 0.874 | 0.591 | 0.562 | 0.455 | 0.666 |

Temporal findings:

- The original trajectory-loss baseline is still the best temporal/balanced model.
- `high_only` gives the best high-severity prediction: high-only `R2log=0.505`, compared with baseline `0.491`.
- Mild high/moderate weighting improves low/moderate decomposition but does not improve high severity.
- Stronger state weighting, stronger cracked-row weighting, higher total loss, and smaller Huber delta did not help.
- Smaller Huber delta was clearly harmful, suggesting that over-robustifying the loss hides useful severity signal.

## Full Validation

I full-validated the two useful candidates from the screen:

- `high_only`: best high-only severity signal.
- `high_mild`: near-baseline temporal fit with gentler severity weighting.

The existing all-pairs trajectory model is used as the full baseline.

| model | temporal total R2log | cold unseen total R2log | first-anchor total R2log | rolling-anchor total R2log |
|---|---:|---:|---:|---:|
| baseline trajectory | 0.798 | 0.155 | 0.497 | 0.735 |
| high_only | 0.789 | 0.200 | 0.517 | 0.749 |
| high_mild | 0.796 | 0.181 | 0.489 | 0.748 |

Severity-specific full-validation behavior:

| model | eval | low-only R2log | mod-only R2log | high-only R2log | onset AUC |
|---|---|---:|---:|---:|---:|
| baseline trajectory | temporal | 0.577 | 0.567 | 0.491 | 0.898 |
| high_only | temporal | 0.571 | 0.556 | 0.505 | 0.866 |
| high_mild | temporal | 0.595 | 0.575 | 0.473 | 0.874 |
| baseline trajectory | cold unseen | 0.051 | 0.081 | -0.028 | 0.757 |
| high_only | cold unseen | 0.081 | 0.112 | -0.016 | 0.751 |
| high_mild | cold unseen | 0.069 | 0.101 | -0.031 | 0.758 |
| baseline trajectory | first anchor | 0.194 | 0.154 | -0.054 | 0.804 |
| high_only | first anchor | 0.202 | 0.191 | -0.016 | 0.809 |
| high_mild | first anchor | 0.177 | 0.121 | -0.032 | 0.813 |
| baseline trajectory | rolling anchor | 0.518 | 0.479 | 0.356 | 0.839 |
| high_only | rolling anchor | 0.533 | 0.483 | 0.365 | 0.843 |
| high_mild | rolling anchor | 0.536 | 0.496 | 0.352 | 0.847 |

## Interpretation

The temporal screen alone says: keep the original trajectory-loss baseline.

The full validation says something more interesting: `high_only` is probably the better practical forecasting model. It loses about `0.009` temporal total `R2log`, but improves:

- cold unseen total `R2log`: `0.155 -> 0.200`
- first-anchor total `R2log`: `0.497 -> 0.517`
- rolling-anchor total `R2log`: `0.735 -> 0.749`
- temporal high-only `R2log`: `0.491 -> 0.505`
- first-anchor high-only `R2log`: `-0.054 -> -0.016`
- rolling-anchor high-only `R2log`: `0.356 -> 0.365`

That tradeoff is aligned with our goal: make the PINN more severity-aware and less dominated by low/no-crack cases.

## Recommendation

Use two named models going forward:

1. **Best temporal reference:** original all-pairs trajectory PINN  
   Loss: low/mod/high band weights `1.0 / 1.2 / 1.5`.

2. **Recommended severity-aware forecasting model:** `high_only`  
   Loss: low/mod/high band weights `0.7 / 1.2 / 3.0`, with `w_total=0.4`, `w_onset=0.15`, `w_cracked=0.5`, `huber_delta=1.0`.

The next highest-value check is seed stability for baseline versus `high_only`, then a narrow sweep around `high_only`, for example:

- low weight: `0.6, 0.7, 0.8`
- moderate weight: `1.1, 1.2, 1.4`
- high weight: `2.5, 3.0, 3.5`

