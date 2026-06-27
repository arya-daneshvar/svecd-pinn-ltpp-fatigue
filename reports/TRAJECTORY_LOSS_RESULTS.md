# Multi-Step Trajectory Loss Results

This test adds an explicit multi-step trajectory loss to the corrected severity-state pure PINN.

The goal was to improve long-horizon anchored prediction, especially the weak first-anchor case, without changing the target definition or adding hybrid residual models.

## Implementation

`train_severity_state_pinn.py` now supports:

```bash
STATE_PINN_W_TRAJ_LIST=...
STATE_PINN_TRAJ_PAIR_MODE=first|all|long
STATE_PINN_TRAJ_MAX_PAIRS=...
```

With `w_traj > 0`, training adds anchored future-prediction pairs:

```text
E_hat(t_future) = E_obs(t_anchor) + relu(E_PINN(t_future | anchor features) - E_PINN(t_anchor | anchor features))
```

The important detail is that the future row's anchor features are rewritten to match the selected anchor row:

- `LAST_OBS_CPCT_L`
- `AGE_AT_LAST_OBS`
- `DT_SINCE_LAST`

This prevents the loss from mixing one anchor definition in the target with another anchor definition in the input features.

## Tested Settings

All trajectory runs used:

- corrected LTPP disjoint severity-state targets
- `feature_cols_sdr_dense_state_obs30.txt`
- L-only anchor
- `w_phys=0.10`
- `w_trans=2`

Trajectory variants:

- `first` pairs: first observation in a section to every later observation
- `all` pairs: every earlier observation to every later observation
- `w_traj = 0.25, 0.5, 1.0`

## Main Results

| model | temporal total `R2log` | cold unseen `R2log` | first-anchor `R2log` | rolling-anchor `R2log` | onset AUC |
|---|---:|---:|---:|---:|---:|
| obs30 fixed | 0.792 | 0.155 | 0.360 | 0.753 | 0.885 |
| first-pair traj `w=0.25` | 0.793 | 0.155 | 0.456 | 0.747 | 0.857 |
| first-pair traj `w=0.5` | 0.778 | 0.155 | 0.474 | 0.749 | 0.862 |
| first-pair traj `w=1.0` | 0.777 | 0.155 | 0.452 | 0.734 | 0.835 |
| all-pair traj `w=0.25` | 0.798 | 0.155 | 0.497 | 0.735 | 0.898 |
| obs30 temporal-tuned | 0.803 | 0.198 | 0.191 | 0.715 | 0.904 |
| full dense SDR | 0.781 | 0.191 | 0.300 | 0.754 | 0.874 |
| obs20 flags | 0.786 | 0.201 | 0.331 | 0.747 | 0.883 |

Severity-specific comparison:

| model | temporal low-only | temporal mod-only | temporal high-only | first low-only | first mod-only | first high-only | rolling low-only | rolling mod-only | rolling high-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| obs30 fixed | 0.606 | 0.566 | 0.468 | 0.221 | 0.078 | -0.050 | 0.546 | 0.477 | 0.352 |
| first-pair traj `w=0.25` | 0.604 | 0.567 | 0.437 | 0.139 | 0.138 | -0.198 | 0.543 | 0.466 | 0.352 |
| first-pair traj `w=0.5` | 0.605 | 0.576 | 0.442 | 0.129 | 0.096 | -0.182 | 0.539 | 0.476 | 0.342 |
| all-pair traj `w=0.25` | 0.577 | 0.567 | 0.491 | 0.194 | 0.154 | -0.054 | 0.518 | 0.479 | 0.356 |

## Interpretation

The trajectory loss worked for the weakness it was designed to address.

The best first-anchor result is now the all-pairs trajectory model:

```text
first-anchor R2log: 0.360 -> 0.497
```

That is a large improvement. It means the model learned a better long-horizon anchored progression from early section observations.

The cost is rolling-anchor performance:

```text
rolling-anchor R2log: 0.753 -> 0.735
```

So the trajectory loss shifts the model toward long-horizon prediction and away from the strongest short-horizon rolling-anchor behavior.

The all-pairs trajectory model also improves temporal total cracking:

```text
temporal R2log: 0.792 -> 0.798
```

and gives the best high-severity temporal score among the compact `obs30` models:

```text
high-only temporal R2log: 0.468 -> 0.491
```

Cold-start unseen-section performance is unchanged, as expected. Trajectory loss cannot solve cold-start because prediction has no section history available.

## Recommendation

Use two named model configurations going forward:

### Balanced Rolling Default

Use when recent distress history is available and rolling prediction is the target.

- Feature list: `feature_cols_sdr_dense_state_obs30.txt`
- `w_phys=0.10`
- `w_trans=2`
- `w_traj=0`
- trajectory mode: none

Performance:

- temporal `R2log = 0.792`
- first-anchor `R2log = 0.360`
- rolling-anchor `R2log = 0.753`

### Early-Anchor Trajectory Model

Use when the goal is prediction from early/limited section history.

- Feature list: `feature_cols_sdr_dense_state_obs30.txt`
- `w_phys=0.10`
- `w_trans=2`
- `w_traj=0.25`
- `STATE_PINN_TRAJ_PAIR_MODE=all`

Performance:

- temporal `R2log = 0.798`
- first-anchor `R2log = 0.497`
- rolling-anchor `R2log = 0.735`

This is the first modification that meaningfully improved the first-anchor scenario while preserving a strong pure-PINN structure.

## Output Folders

- `outputs_severity_state_pinn_sdr_dense_state_obs30_traj_first025_full`
- `outputs_severity_state_pinn_sdr_dense_state_obs30_traj_first05_full`
- `outputs_severity_state_pinn_sdr_dense_state_obs30_traj_first1_full`
- `outputs_severity_state_pinn_sdr_dense_state_obs30_traj_all025_full`
- `outputs_feature_pruning_state/trajectory_loss_comparison.csv`
