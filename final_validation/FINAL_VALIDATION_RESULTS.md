# Final validation results

## Methodology

The frozen pure S-VECD-informed severity-state PINN was evaluated without changing its architecture,
features, loss weights, targets, splits, anchor rewriting, or seeds. Preprocessing was corrected so
that dense-field completion, median imputation, and scaling use training rows only. The model-ready
dataset contains 2,285 rows from 326 sections. Targets are disjoint low-, moderate-, and high-severity
cracking area percentages; total cracking is their sum. The exact headline metric is R² on `log1p`
total cracking. Onset is observed total cracking >0%; the PINN probability is its frozen BCE-trained
onset head. Split-specific transformations were applied unchanged to held-out rows.

Uncertainty used 2,000 paired percentile bootstrap replicates (seed 20260714), sampling
sections with replacement and retaining every row with cluster multiplicity. All paired comparisons
used identical row IDs. Reliability subgroup intervals used 1,000 section-cluster replicates.

Robustness used five fixed corruption seeds shared across models. Training-label Gaussian noise had
SD 1, 3, and 5 percentage points; corrupted disjoint states were clipped to [0,100] and proportionally
rescaled when their sum exceeded 100. Inference tests used 1/3/5-point anchor noise, section-block
feature masking at 25/50/75/100%, and a prespecified combined case (3-point anchor noise plus 50%
traffic and 50% climate masking). Clean-training imputers were retained for inference masking.

## Predictive accuracy and statistical uncertainty

Frozen PINN total R²log was 0.788 (95% CI 0.687–0.862)
for temporal anchoring, 0.188 (0.101–0.257) for cold start,
0.491 (0.385–0.581) for first-anchor, and
0.741 (0.693–0.782) for rolling-anchor prediction.

Against XGBoost, PINN-oriented paired R²log differences were 0.036
(-0.008–0.070) temporally, -0.253
(-0.342–-0.163) at cold start, -0.080
(-0.179–0.010) for first anchor, and 0.020
(-0.021–0.069) for rolling anchor. Thus the temporal, first-anchor, and
rolling-anchor R²log differences versus XGBoost did not exclude zero; XGBoost was clearly better at
cold start. The PINN exceeded anchored persistence for temporal and rolling R²log with paired
intervals excluding zero. The secondary residual hybrid achieved a small temporal R²log improvement
but worse raw MAE, so it does not replace the pure PINN result.

## Reduced and degraded data robustness

The row-level rerun confirmed the earlier limited-section pattern: at 100%, 50%, 20%, and 10%
training sections, mean PINN R²log values were 0.788, 0.766, 0.751, and 0.746; corresponding
XGBoost values were 0.750, 0.691, 0.580, and 0.457. Complete distributions and section intervals are reported in
`outputs_bootstrap_uncertainty`. This is supportive evidence about anchored low-data learning, not
cold-start generalization.

Under severe (5-point SD) training-label noise, mean R²log was 0.349
for the PINN versus 0.260 for XGBoost; at moderate noise the values were
0.429 and 0.459. The PINN
therefore showed an advantage only at the severe level; it was not best at moderate noise. Complete
removal of traffic features reduced PINN R²log to approximately 0.749, while full climate,
mechanistic, and structure/material masking produced smaller changes. Anchor-feature removal in the
temporal implementation is labeled a scenario conversion and is not interpreted as ordinary
anchored forecasting.

The combined case was unfavorable to the PINN: mean R²log was
0.287, compared with
0.447 for XGBoost. Moderate and severe anchor noise drove
R²log below zero for all learned models; XGBoost degraded less. Accordingly, the frozen PINN is not
uniformly more robust to noisy or missing operational data.

## Calibration and reliability

First-anchor PINN MAE increased from 1.335% at ≤2 years to
4.212% at 6–10 years; high-severity MAE was largest beyond
10 years (2.309%). By observed condition, first-anchor total
MAE was 0.855% for no cracking,
8.133% for moderate-dominant, and
9.676% for high-dominant observations. Mean signed
error was negative in cracked severity groups, demonstrating systematic underprediction.

First-anchor onset calibration was weak for the PINN (Brier 0.254, ECE 0.235,
slope 0.222) compared with XGBoost (Brier 0.123, ECE
0.020, slope 1.029). Among 79 high-dominant
first-anchor observations, the PINN underpredicted the high state in 92.4%
and had a 9.7% rate of absolute total error >10 percentage points.

## Negative findings and limitations

- Conventional GBMs were stronger for cold-start unseen sections.
- PINN advantages over GBMs in the principal anchored R²log comparisons were mostly uncertain.
- Robustness was degradation-specific; combined and anchor-noise results weakened broad claims.
- The PINN onset head was materially miscalibrated and should not be used as a decision probability
  without training-only calibration in future work.
- High-severity cases were sparse (79 first-anchor rows from 37 sections) and systematically underpredicted.
- The frozen cold-start routine does not activate trajectory pairs despite the variant label; this
  discrepancy was preserved for exact reproduction and limits interpretation of cold-start physics.
- Results apply to these LTPP sections and protocols, not nationwide generalization.
