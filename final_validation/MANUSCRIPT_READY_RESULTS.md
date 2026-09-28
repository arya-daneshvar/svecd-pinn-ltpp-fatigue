# Manuscript-ready results

## Methods — Additional robustness and reliability analyses

Sampling uncertainty was quantified using a paired cluster bootstrap in which pavement sections,
rather than individual surveys, were sampled with replacement. We used 2,000 replicates and
percentile 95% confidence intervals; the same sampled sections and all surveys within them were used
for every paired model comparison. The frozen PINN and conventional comparators were evaluated on
identical observations under temporal, cold-start, first-anchor, and rolling-anchor protocols.
Training-data robustness was assessed using prespecified 1-, 3-, and 5-percentage-point Gaussian
noise applied to the three disjoint severity targets. Operational robustness was assessed using
shared anchor-noise realizations, section-block feature-group masking, and a combined scenario with
3-point anchor noise and 50% traffic and climate masking. Reliability was stratified by elapsed time
since anchor, surveys ahead, observed dominant severity (higher severity resolved ties), and anchor
condition. Onset was total cracking >0%; calibration used ten equal-frequency bins, with equal-width
bins as sensitivity analysis. Subgroup intervals used section-cluster resampling.

## Results

The frozen PINN achieved temporal, cold-start, first-anchor, and rolling-anchor total-cracking R²log
values of 0.788, 0.188, 0.491, and 0.741,
respectively. Its temporal paired advantage over XGBoost was 0.036 (95% CI
-0.008 to 0.070), and the models were not distinguishable on this metric.
XGBoost was better at cold start (PINN-oriented difference -0.253, 95% CI
-0.342 to -0.163). The PINN retained more R²log than boosting under severe
training-label noise, although it was not consistently best under moderate noise, anchor noise, or
the combined degradation. Performance deteriorated with first-anchor horizon: PINN MAE rose from
1.335% within 2 years to
4.212% at 6–10 years. Error was highest for
high-dominant observations (9.676% MAE), and
high-severity underprediction was frequent. PINN onset calibration was weaker than XGBoost
(first-anchor Brier 0.254 versus 0.123).

## Discussion

These analyses refine rather than uniformly strengthen the physics-informed modeling claim. The
frozen PINN was competitive and stable in anchored forecasting and under severe label corruption,
but confidence intervals did not establish a general advantage over boosting, and conventional
models were stronger for cold-start and several operational degradations. Increasing error with
horizon, underprediction of high-severity cracking, and weak onset calibration identify conditions
where predictions should be accompanied by monitoring or conservative engineering review. Claims
should therefore be limited to competitive anchored accuracy and degradation-specific robustness,
not universal superiority, immunity to missing data, or broad geographic generalization.

## Proposed figure captions

1. **Main uncertainty figure.** Section-clustered point estimates and percentile 95% confidence intervals for total-cracking R²log across frozen validation protocols.
2. **Main robustness figure.** Absolute total-cracking performance and retention from the clean condition under prespecified data degradation; identical corruption realizations were used across models.
3. **Main reliability figure.** Total and high-severity error versus first-anchor horizon, onset calibration, and error by observed severity for the frozen PINN, XGBoost, and anchored persistence.
4. **Supplementary paired-difference figure.** Paired section-bootstrap differences oriented so positive values favor the pure frozen PINN; the vertical line denotes no difference.
5. **Supplementary limited-data figure.** Total-cracking R²log with section-bootstrap intervals versus retained training-section fraction.
6. **Supplementary label-noise figure.** Performance under training-only severity-label corruption across five shared realizations.
7. **Supplementary anchor-noise figure.** First- and rolling-anchor performance under shared survey-error realizations.
8. **Supplementary feature-missingness figure.** Performance under section-block masking by documented feature group.
9. **Supplementary combined-degradation figure.** Performance under the prespecified combined operational scenario.
10. **Supplementary residual figure.** Total and high-severity residual diagnostics emphasizing large-value underprediction.

## Proposed table captions

1. **Main Table 1.** Frozen PINN performance and section-clustered 95% confidence intervals under four validation protocols, with paired differences against prespecified comparators.
2. **Main Table 2.** Compact operational findings by horizon, observed severity, onset calibration, and catastrophic-error frequency.
3. **Supplementary Table S1.** Complete bootstrap metric summaries and invalid-replicate frequencies.
4. **Supplementary Table S2.** Paired model-difference distributions and empirical comparison probabilities.
5. **Supplementary Table S3.** All reduced-training-section results by fixed subset and repeat.
6. **Supplementary Table S4.** All degradation realizations, absolute changes, relative retention, slopes, and normalized degradation AUC.
7. **Supplementary Table S5.** Horizon, survey-step, observed-severity, anchor-condition, and total-cracking subgroup metrics with sample counts.
8. **Supplementary Table S6.** Equal-frequency and equal-width onset-calibration bins and calibration statistics.
