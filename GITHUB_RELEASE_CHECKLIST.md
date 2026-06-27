# GitHub Release Checklist

Before making the repository public and citing it in the article:

- Replace `REPLACE_WITH_FAMILY_NAME` in `CITATION.cff`.
- Replace `REPLACE_WITH_GITHUB_URL` in `CITATION.cff`.
- Replace `REPLACE_WITH_ARTICLE_TITLE` in `CITATION.cff`, or remove the `preferred-citation` block until the title is final.
- Confirm the code license holder name in `LICENSE`.
- Create a GitHub release, for example `v1.0.0`.
- Archive the release with Zenodo or OSF to obtain a DOI.
- Add the DOI to `CITATION.cff` and the article.
- Cite FHWA LTPP / InfoPave separately as the original data source.

Suggested article wording:

```text
The frozen PINN implementation, processed LTPP-derived dataset, feature list,
and reproducibility scripts are archived at [repository DOI]. Original pavement
performance data were obtained from the FHWA Long-Term Pavement Performance
program through LTPP InfoPave / Standard Data Release resources.
```

