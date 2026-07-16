# GitHub Release Checklist

Before making the repository public and citing it in the article:

- Confirm the code license holder name in `LICENSE`.
- Confirm the article title in the `preferred-citation` block of `CITATION.cff`.
- Run `python3 check_artifact.py`.
- Run `python3 run_frozen_best.py --parts temporal --suffix _release_temporal_check`.
- Compare the generated temporal metrics with `ARTIFACT_MANIFEST.md`.
- Confirm that `final_validation/final_validation_predictions.csv`,
  `limited_data_predictions.csv`, summary CSVs, and publication figures are
  present.
- Confirm that no file exceeds GitHub's 100 MB per-file limit.
- Create a GitHub release, for example `v1.0.0`.
- Archive the release with Zenodo or OSF to obtain a DOI.
- Add the DOI to `CITATION.cff` and the article.
- Cite FHWA LTPP / InfoPave separately as the original data source.

Suggested article wording:

```text
The frozen PINN implementation, processed LTPP-derived dataset, feature list,
row-level out-of-sample predictions, evaluation outputs, and reproducibility
scripts are archived at [repository DOI]. Original pavement-performance data
were obtained from the FHWA Long-Term Pavement Performance program through LTPP
InfoPave / Standard Data Release resources. The artifact contains a frozen
retraining recipe rather than a serialized neural-network checkpoint.
```
