# Split-local preprocessing audit

## Correction

The frozen runner now reconstructs every selected predictor that has a corresponding
`STRICT_` source column independently inside each model fit. Missing values are filled
from training rows only, in this order: section median, state-year median, state median,
training-set global median, and finally zero only when no finite training value exists.
The downstream median imputer and standard scaler are also fitted only on training rows.
The fitted transformations are then applied unchanged to validation or test rows.

Splits are established before this reconstruction. Model fitting, trajectory-pair
construction, anchor rewriting, and all learned preprocessing therefore exclude the
corresponding test rows. The same rule is active by default in `run_frozen_best.py` and
the row-level validation utilities.

## Controlled comparison

The uncorrected control reproduced the archived prediction files byte for byte. Replacing
only the cross-split dense-field reconstruction changed the seed-0 results as follows:

| protocol | archived R2log | split-local R2log | difference |
|---|---:|---:|---:|
| temporal latest-survey | 0.788747 | 0.787721 | -0.001026 |
| cold-start unseen sections | 0.199996 | 0.188096 | -0.011899 |
| first-anchor unseen sections | 0.516774 | 0.491478 | -0.025296 |
| rolling-anchor unseen sections | 0.749353 | 0.741278 | -0.008074 |

For the temporal test, raw-scale R2 increased from 0.726874 to 0.733774, RMSE decreased
from 6.892249 to 6.804631 percentage points, and MAE decreased from 2.984303 to 2.929282
percentage points. Across five seeds, corrected mean R2log values were 0.776037
(temporal), 0.182050 (cold start), 0.492117 (first anchor), and 0.739707 (rolling
anchor). The row-level corrected predictions and 2,000-replicate section-clustered
bootstrap summaries are stored in this directory.

## Scope of the verification

This audit establishes split-local preprocessing and fitting. It does not convert all
recorded predictor histories into a real-time deployment simulation. Static design and
section descriptors are treated as fixed covariates, while target-period traffic,
climate, and mechanistic quantities must be supplied as prescribed forecasts or
scenarios for a true prospective prediction. Evaluations that use their recorded values
are retrospective conditional forecasts. The manuscript states this boundary explicitly.
