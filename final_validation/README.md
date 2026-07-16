# Final validation of the frozen S-VECD-informed PINN

This folder contains post-freeze validation only. It imports the locked
architecture, loss, features, targets, split conventions, preprocessing, and
anchor logic from the repository root. No model or physics weight is selected
from validation/test performance.

## Commands

```bash
python3 run_all_final_validations.py --smoke-test
python3 run_all_final_validations.py --resume
```

Individual packages accept `--smoke-test` and `--resume`:

```bash
python3 run_bootstrap_uncertainty.py --resume --bootstrap-replicates 2000
python3 run_data_robustness.py --resume --seeds 0,1,2,3,4
python3 run_reliability_analysis.py --resume --bootstrap-replicates 1000
python3 plot_final_validation.py
```

The smoke test executes a full frozen temporal fit so the recorded temporal metric can be checked;
other packages use reduced repetitions/folds. Full results are written incrementally and prior
valid row-level predictions are reused by `--resume`.

The onset definition is total observed cracking greater than 0%. The frozen neural onset head,
which is explicitly trained with binary cross-entropy in the frozen loss, supplies PINN onset
probabilities. Conventional models use a prespecified training-only classifier. No probability
mapping is fitted on test outcomes.

## Outputs

- `outputs_bootstrap_uncertainty/`: paired section-cluster bootstrap and limited-data evidence.
- `outputs_data_robustness/`: training-label, anchor-noise, group-missingness, and combined tests.
- `outputs_reliability/`: horizon/severity reliability, calibration, and residual diagnostics.
- `final_validation_predictions.csv`: row-level out-of-sample predictions used by uncertainty and reliability analyses.
- `limited_data_predictions.csv`: fixed-subset row predictions used for limited-data intervals.

All PNG figures are exported at 600 dpi and paired with vector PDF files.

The two raw bootstrap-replicate matrices (`bootstrap_metric_distributions.csv`
and `paired_model_differences.csv`) are intentionally omitted from the public
bundle because together they exceed 240 MB. Their configuration, invalid-run
log, sample manifest, compact metric summaries, paired confidence intervals,
and publication figures are included.

No neural-network checkpoint is present. The primary model is regenerated from
the frozen repository-root recipe; archived predictions allow metric and figure
verification without retraining.
