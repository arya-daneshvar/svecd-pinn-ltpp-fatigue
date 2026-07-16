# Data Notice

This repository contains processed pavement-performance data derived from the Federal Highway Administration Long-Term Pavement Performance program, accessed through LTPP InfoPave / Standard Data Release resources.

## Source Attribution

Original source:

- Federal Highway Administration, Long-Term Pavement Performance program
- LTPP InfoPave: https://infopave.fhwa.dot.gov/

The processed files in this repository were created for research reproducibility and should be cited together with the original FHWA LTPP / InfoPave data source.

## Processed Data Included Here

The repository includes:

- `fatigue_svecd_sdr.csv`
- `fatigue_svecd_sdr_dense.csv`
- `fatigue_svecd_sdr_dense_state.csv`
- `data_cache/ANALYSIS_DIS_AC.csv`
- feature-list and data-dictionary files used by the frozen model
- `final_validation/final_validation_predictions.csv`
- `final_validation/limited_data_predictions.csv`

The final model dataset is `fatigue_svecd_sdr_dense_state.csv`.

## License Boundary

The MIT license in this repository applies to the code written for the modeling and reproducibility workflow. It does not assert ownership over FHWA/LTPP source data.

Users should follow applicable FHWA/LTPP/InfoPave terms, attribution expectations, and citation practices when reusing the processed data.

## Reproducibility Notes

The full raw SDR cache is not included. The frozen generated model dataset is included, and `data_cache/ANALYSIS_DIS_AC.csv` is included to make the final severity-state target construction auditable.

The archived prediction files contain derived model outputs and row identifiers
needed for verification; they do not replace or redistribute the complete raw
LTPP Standard Data Release.
