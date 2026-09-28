#!/usr/bin/env python3
"""Generate evidence-linked scientific and manuscript-ready reports from saved CSVs."""
from __future__ import annotations
import pandas as pd
import numpy as np
from validation_utils import *

def fmt(x,n=3): return "undefined" if pd.isna(x) else f"{x:.{n}f}"
def paired(protocol,comp,metric="R2log_total"):
 p=pd.read_csv(OUT_BOOT/"paired_model_differences_summary.csv")
 return p[(p.sample_type=="protocol")&(p.protocol==protocol)&(p.comparator==comp)&(p.metric==metric)].iloc[0]
def metric(protocol,model,metric="R2log_total"):
 s=pd.read_csv(OUT_BOOT/"bootstrap_metric_summary.csv")
 return s[(s.sample_type=="protocol")&(s.protocol==protocol)&(s.model==model)&(s.metric==metric)].iloc[0]

def main():
 sboot=pd.read_csv(OUT_BOOT/"bootstrap_metric_summary.csv"); robust=pd.read_csv(OUT_ROBUST/"robustness_summary.csv")
 ret=pd.read_csv(OUT_ROBUST/"robustness_retention_summary.csv"); h=pd.read_csv(OUT_REL/"horizon_metrics.csv")
 sev=pd.read_csv(OUT_REL/"severity_stratified_metrics.csv"); cal=pd.read_csv(OUT_REL/"onset_calibration_metrics.csv"); high=pd.read_csv(OUT_REL/"high_severity_error_summary.csv")
 temporal=metric("temporal","Frozen pure PINN"); cold=metric("cold_start","Frozen pure PINN"); first=metric("first_anchor","Frozen pure PINN"); roll=metric("rolling_anchor","Frozen pure PINN")
 px=paired("temporal","XGBoost"); cx=paired("cold_start","XGBoost"); fx=paired("first_anchor","XGBoost"); rx=paired("rolling_anchor","XGBoost")
 lab=robust[(robust.degradation_family=="training_label_noise")&(robust.metric=="R2log_total")]
 def lv(level,model): return lab[(lab.level==level)&(lab.model==model)].iloc[0]
 comb=robust[(robust.degradation_family=="combined")&(robust.metric=="R2log_total")]
 hp=h[(h.protocol=="first_anchor")&(h.model=="Frozen pure PINN")&(h.stratification=="horizon")]
 sp=sev[(sev.protocol=="first_anchor")&(sev.model=="Frozen pure PINN")&(sev.stratification=="observed_severity")]
 cp=cal[(cal.protocol=="first_anchor")&(cal.model=="Frozen pure PINN")].iloc[0]; cxcal=cal[(cal.protocol=="first_anchor")&(cal.model=="XGBoost")].iloc[0]
 hip=high[(high.protocol=="first_anchor")&(high.model=="Frozen pure PINN")].iloc[0]

 results=f"""# Final validation results

## Methodology

The frozen pure S-VECD-informed severity-state PINN was evaluated without changing its architecture,
features, loss weights, targets, splits, anchor rewriting, or seeds. Preprocessing was corrected so
that dense-field completion, median imputation, and scaling use training rows only. The model-ready
dataset contains 2,285 rows from 326 sections. Targets are disjoint low-, moderate-, and high-severity
cracking area percentages; total cracking is their sum. The exact headline metric is R² on `log1p`
total cracking. Onset is observed total cracking >0%; the PINN probability is its frozen BCE-trained
onset head. Split-specific transformations were applied unchanged to held-out rows.

Uncertainty used 2,000 paired percentile bootstrap replicates (seed {BOOTSTRAP_SEED}), sampling
sections with replacement and retaining every row with cluster multiplicity. All paired comparisons
used identical row IDs. Reliability subgroup intervals used 1,000 section-cluster replicates.

Robustness used five fixed corruption seeds shared across models. Training-label Gaussian noise had
SD 1, 3, and 5 percentage points; corrupted disjoint states were clipped to [0,100] and proportionally
rescaled when their sum exceeded 100. Inference tests used 1/3/5-point anchor noise, section-block
feature masking at 25/50/75/100%, and a prespecified combined case (3-point anchor noise plus 50%
traffic and 50% climate masking). Clean-training imputers were retained for inference masking.

## Predictive accuracy and statistical uncertainty

Frozen PINN total R²log was {fmt(temporal.observed)} (95% CI {fmt(temporal.ci_lower)}–{fmt(temporal.ci_upper)})
for temporal anchoring, {fmt(cold.observed)} ({fmt(cold.ci_lower)}–{fmt(cold.ci_upper)}) for cold start,
{fmt(first.observed)} ({fmt(first.ci_lower)}–{fmt(first.ci_upper)}) for first-anchor, and
{fmt(roll.observed)} ({fmt(roll.ci_lower)}–{fmt(roll.ci_upper)}) for rolling-anchor prediction.

Against XGBoost, PINN-oriented paired R²log differences were {fmt(px.observed_difference)}
({fmt(px.ci_lower)}–{fmt(px.ci_upper)}) temporally, {fmt(cx.observed_difference)}
({fmt(cx.ci_lower)}–{fmt(cx.ci_upper)}) at cold start, {fmt(fx.observed_difference)}
({fmt(fx.ci_lower)}–{fmt(fx.ci_upper)}) for first anchor, and {fmt(rx.observed_difference)}
({fmt(rx.ci_lower)}–{fmt(rx.ci_upper)}) for rolling anchor. Thus the temporal, first-anchor, and
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

Under severe (5-point SD) training-label noise, mean R²log was {fmt(lv('severe','Frozen pure PINN')['mean'])}
for the PINN versus {fmt(lv('severe','XGBoost')['mean'])} for XGBoost; at moderate noise the values were
{fmt(lv('moderate','Frozen pure PINN')['mean'])} and {fmt(lv('moderate','XGBoost')['mean'])}. The PINN
therefore showed an advantage only at the severe level; it was not best at moderate noise. Complete
removal of traffic features reduced PINN R²log to approximately 0.749, while full climate,
mechanistic, and structure/material masking produced smaller changes. Anchor-feature removal in the
temporal implementation is labeled a scenario conversion and is not interpreted as ordinary
anchored forecasting.

The combined case was unfavorable to the PINN: mean R²log was
{fmt(comb[comb.model=='Frozen pure PINN']['mean'].iloc[0])}, compared with
{fmt(comb[comb.model=='XGBoost']['mean'].iloc[0])} for XGBoost. Moderate and severe anchor noise drove
R²log below zero for all learned models; XGBoost degraded less. Accordingly, the frozen PINN is not
uniformly more robust to noisy or missing operational data.

## Calibration and reliability

First-anchor PINN MAE increased from {fmt(hp[hp.subgroup=='<=2 y'].MAE_total.iloc[0])}% at ≤2 years to
{fmt(hp[hp.subgroup=='>6-10 y'].MAE_total.iloc[0])}% at 6–10 years; high-severity MAE was largest beyond
10 years ({fmt(hp[hp.subgroup=='>10 y'].MAE_HIGH.iloc[0])}%). By observed condition, first-anchor total
MAE was {fmt(sp[sp.subgroup=='no cracking'].MAE_total.iloc[0])}% for no cracking,
{fmt(sp[sp.subgroup=='moderate-dominant'].MAE_total.iloc[0])}% for moderate-dominant, and
{fmt(sp[sp.subgroup=='high-dominant'].MAE_total.iloc[0])}% for high-dominant observations. Mean signed
error was negative in cracked severity groups, demonstrating systematic underprediction.

First-anchor onset calibration was weak for the PINN (Brier {fmt(cp.Brier)}, ECE {fmt(cp.ECE_equal_frequency)},
slope {fmt(cp.calibration_slope)}) compared with XGBoost (Brier {fmt(cxcal.Brier)}, ECE
{fmt(cxcal.ECE_equal_frequency)}, slope {fmt(cxcal.calibration_slope)}). Among 79 high-dominant
first-anchor observations, the PINN underpredicted the high state in {100*hip.high_underprediction_fraction:.1f}%
and had a {100*hip.catastrophic_error_rate:.1f}% rate of absolute total error >10 percentage points.

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
"""
 (HERE/"FINAL_VALIDATION_RESULTS.md").write_text(results)

 manuscript=f"""# Manuscript-ready results

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
values of {fmt(temporal.observed)}, {fmt(cold.observed)}, {fmt(first.observed)}, and {fmt(roll.observed)},
respectively. Its temporal paired advantage over XGBoost was {fmt(px.observed_difference)} (95% CI
{fmt(px.ci_lower)} to {fmt(px.ci_upper)}), and the models were not distinguishable on this metric.
XGBoost was better at cold start (PINN-oriented difference {fmt(cx.observed_difference)}, 95% CI
{fmt(cx.ci_lower)} to {fmt(cx.ci_upper)}). The PINN retained more R²log than boosting under severe
training-label noise, although it was not consistently best under moderate noise, anchor noise, or
the combined degradation. Performance deteriorated with first-anchor horizon: PINN MAE rose from
{fmt(hp[hp.subgroup=='<=2 y'].MAE_total.iloc[0])}% within 2 years to
{fmt(hp[hp.subgroup=='>6-10 y'].MAE_total.iloc[0])}% at 6–10 years. Error was highest for
high-dominant observations ({fmt(sp[sp.subgroup=='high-dominant'].MAE_total.iloc[0])}% MAE), and
high-severity underprediction was frequent. PINN onset calibration was weaker than XGBoost
(first-anchor Brier {fmt(cp.Brier)} versus {fmt(cxcal.Brier)}).

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
"""
 (HERE/"MANUSCRIPT_READY_RESULTS.md").write_text(manuscript)

 executive=f"""# Final validation executive summary

1. **Do confidence intervals support the paper's main comparisons?** They support the PINN's
   advantage over persistence for temporal and rolling anchored R²log. They do not support a clear
   temporal/first/rolling advantage over XGBoost; those paired intervals include zero. At cold start,
   the interval supports the opposite conclusion: XGBoost is better.

2. **Is the PINN genuinely more robust?** Only conditionally. It retained more performance at severe
   training-label noise and under reduced training sections, but was not consistently best at
   moderate noise, anchor noise, or the combined degradation. A universal robustness claim is not supported.

3. **When is it most and least reliable?** It is most reliable for short (≤2-year), low/no-cracking
   anchored forecasts and rolling updates. It is least reliable at long first-anchor horizons,
   moderate/high observed severity, and noisy anchors. High-severity underprediction is common and
   the onset head is poorly calibrated.

4. **Main manuscript:** one protocol/paired-CI table; the absolute-versus-relative robustness figure;
   the four-panel reliability summary; and a compact operational table with horizon, severity,
   calibration, and catastrophic-error findings.

5. **Supplement:** full bootstrap distributions, every comparator/metric, fixed limited-data subsets,
   all corruption seeds and groups, invalid replicates, complete subgroup tables, equal-width
   calibration, and residual diagnostics.

6. **Claims weakened:** broad claims of superiority to GBMs, universal missing/noisy-data robustness,
   reliable cold-start generalization, calibrated onset probability, and reliable high-severity
   forecasting should be weakened or removed.

7. **Defensible claims:** the frozen pure PINN provides competitive anchored total-cracking accuracy;
   exceeds persistence in key anchored R²log comparisons; retains anchored performance well when
   training sections are reduced; can outperform boosting under severe training-label corruption;
   deteriorates with forecast horizon; and requires caution for cold-start, noisy-anchor,
   high-severity, and probability-calibration use.
"""
 (HERE/"FINAL_VALIDATION_EXECUTIVE_SUMMARY.md").write_text(executive)

if __name__=="__main__": main()
