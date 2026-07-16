# Final validation repository audit

## Frozen reference and implementation

The mandatory reference is reproducible as a recipe, not a serialized checkpoint. The wrapper
`Codex - Frozen/run_frozen_best.py` locks the dense severity-state dataset, obs30 feature list,
pure (non-hybrid) PINN, severity-aware `high_only` loss, and
`state_transition_w0.1_t2_traj0.25` configuration. `train_severity_state_pinn.py` fits the median
imputer and standard scaler only on training rows. Its temporal split holds out the latest
progression survey for each eligible section. Cold and warm validation use five-fold GroupKFold
by `SECTION_ID`; first-anchor uses the first held-out survey, and rolling-anchor uses the immediately
preceding held-out survey.

The exact project total/log metric is R² applied to `log1p(observed total)` and
`log1p(predicted total)`. Onset is defined as total observed cracking greater than 0%.

## Existing predictions and metrics

The principal frozen output folder contains aggregate CSVs only:

- `temporal_state_sweep.csv`: three aggregate rows; selected temporal transition R²log = 0.7887469.
- `scenario_a_state_unseen_sections.csv`: one aggregate cold-start row; R²log = 0.1999956.
- `warm_start_state_unseen_sections.csv`: aggregate first-anchor and rolling-anchor rows;
  R²log = 0.5167744 and 0.7493527.
- seed-stability outputs contain per-seed aggregate metrics, not row predictions.

Consequently, the frozen CSVs do **not** contain section ID, survey date/order, observed state
targets, predicted state targets, total cracking, anchor survey, or forecast horizon. Those fields
cannot be recovered from aggregate metrics; row-level out-of-sample predictions must be newly
executed for section bootstrap and reliability diagnostics.

An existing PINN-residual-hybrid prediction file contains section ID, survey date, observed and
predicted low/moderate/high states and totals, but no anchor survey or horizon. It comes from a
different PINN implementation/configuration and a 277-row temporal subset (831 stacked model rows),
so it is not paired-comparable with the frozen reference and is excluded. A newly evaluated hybrid
is permitted only as a clearly secondary temporal comparison on identical rows.

The validation-protocol audit stores only aggregate protocol metrics. Earlier pure-PINN warm/k-shot
scripts sometimes store total-band row predictions, but they precede the frozen severity-aware
model and omit all three disjoint predicted states; they are not valid substitutes.

## Available conventional models

The mainstream limited-data script already defines XGBoost, HistGradientBoosting, random forest,
extra trees, ridge, MLP, and anchored persistence on the exact frozen dataset/features and temporal
test rows. CatBoost is installed but was not part of that completed run. Its output has aggregate
metrics only. The fixed 100%, 50%, and 10% section subsets can be reconstructed exactly from the
published seed rule; the requested 20% subset is absent (the prior run used 25%).

## Completed degradation experiments

`Codex - LimitedData/outputs_limited_data` contains 1,026 evaluations spanning section reduction,
label/onset/physics-input noise, feature removal, and a combined scenario. It is not a frozen-model
comparison: it uses a different `_mhB` cumulative-target model/dataset, 250 epochs, and a six-value
physics-weight grid, then reports the best weight separately for each degradation case. It therefore
cannot support claims about the frozen pure PINN and will not be repeated or treated as central
evidence. Its numerical levels inform the prespecified 1/3/5 percentage-point label-noise design.

`Codex - MainstreamCompare/outputs_mainstream_limited_data` is configuration-compatible and already
completed 104 aggregate evaluations. Its fixed subsets and aggregate values are reusable as an
independent cross-check, but clustered intervals require new row predictions.

## Analyses possible without retraining

- Audit and independently recompute recorded aggregate metrics only where row predictions exist.
- Reliability analysis after one valid row-level prediction-generation pass; subsequent horizon,
  severity, residual, calibration, and bootstrap work is post-processing.
- Anchor-noise and inference-missingness realizations after clean models/preprocessors are fitted;
  no refitting is required for the PINN in those inference-only scenarios.
- Persistence predictions are deterministic post-processing of observed anchors.

## Experiments requiring execution

- Frozen row predictions for temporal, cold-start, first-anchor, and rolling-anchor protocols.
- Identically sampled XGBoost, CatBoost, HistGradientBoosting, and persistence predictions.
- A compatible secondary temporal PINN + CatBoost residual hybrid.
- Row predictions for fixed 100%, 50%, 20%, and 10% training-section scenarios.
- Frozen-model fits under training-label noise, because corruption changes the fitting targets.
- Clean fits followed by shared inference-time anchor corruption and feature masking.

## Leakage and comparability findings

1. The frozen preprocessing is training-only; no prohibited section overlap occurs in GroupKFold.
2. The prior 1,026-run sweep selected physics weight by degradation/test outcome and is not a fixed
   frozen-model robustness analysis.
3. The existing hybrid predictions are implementation-incompatible and cannot be paired.
4. The frozen cold-start function labels the variant `traj0.25` but does not construct/pass
   trajectory data, so the trajectory loss is effectively inactive in cold-start folds. This is a
   reproducibility discrepancy, not target leakage. Final validation preserves the recorded behavior
   rather than silently changing the frozen artifact.
5. CatBoost and other conventional models are compared only when their row IDs exactly equal the
   frozen model's row IDs. Metrics from different test sets are never paired.
6. Complete anchor removal is labeled a cold-start conversion, not ordinary anchored forecasting.
