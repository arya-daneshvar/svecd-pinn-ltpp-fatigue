# Final validation executive summary

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
