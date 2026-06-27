"""
Build an aggressive/dense SDR feature dataset.

This starts from an existing SDR-tagged dataset, usually fatigue_svecd_sdr.csv, and keeps the
same model-facing SDR feature names while filling missing values with increasingly broad fallback
tiers.  The strict observed values are preserved as STRICT_<feature> audit columns.

Fallback tiers:
  0 observed directly in the strict SDR extraction
  1 median from the same physical section / fatigue segment
  2 median from the same state and survey year
  3 median from the same state
  4 global median
  5 zero fallback, used only if a feature has no finite values at all

The model feature list includes the densified SDR columns plus HAS_<feature> and TIER_<feature>
signals so the PINN can learn when a value is measured versus borrowed.

Run:
  python3 build_sdr_dense_feature_dataset.py

Optional:
  DENSE_SOURCE_TAG=_sdr_mr DENSE_OUT_TAG=_sdr_mr_dense python3 build_sdr_dense_feature_dataset.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


warnings.simplefilter("ignore", PerformanceWarning)


PROJ = Path(__file__).resolve().parent
SOURCE_TAG = os.environ.get("DENSE_SOURCE_TAG", "_sdr")
OUT_TAG = os.environ.get("DENSE_OUT_TAG", f"{SOURCE_TAG}_dense")
IN_CSV = PROJ / f"fatigue_svecd{SOURCE_TAG}.csv"
IN_FEATURES = PROJ / f"feature_cols{SOURCE_TAG}.txt"
OUT_CSV = PROJ / f"fatigue_svecd{OUT_TAG}.csv"
OUT_PARQUET = PROJ / f"fatigue_svecd{OUT_TAG}.parquet"
OUT_FEATURES = PROJ / f"feature_cols{OUT_TAG}.txt"
OUT_DICT = PROJ / f"data_dictionary{OUT_TAG}.md"
OUTDIR = PROJ / f"outputs_sdr_dense{OUT_TAG}"
OUTDIR.mkdir(exist_ok=True)

SDR_PREFIXES = (
    "CLM_",
    "SMP_",
    "TRF_MC_",
    "TRF_GVW_",
    "DEICE_",
    "GPR_",
    "DRAIN_",
    "DLR_",
)


def finite_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def survey_year(df: pd.DataFrame) -> pd.Series:
    if "SURVEY_YEAR" in df.columns:
        return pd.to_numeric(df["SURVEY_YEAR"], errors="coerce")
    return pd.to_datetime(df["SURVEY_DATE"], errors="coerce").dt.year


def section_key(df: pd.DataFrame) -> pd.Series:
    if "PHYSICAL_SECTION_ID" in df.columns:
        return df["PHYSICAL_SECTION_ID"].astype(str)
    return df["SECTION_ID"].astype(str)


def densify_feature(df: pd.DataFrame, feature: str) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    original = finite_numeric(df[feature])
    dense = original.copy()
    tier = pd.Series(np.where(original.notna(), 0, -1), index=df.index, dtype="int16")

    sec = section_key(df)
    sy = survey_year(df)
    state = pd.to_numeric(df["STATE_CODE"], errors="coerce")
    work = pd.DataFrame({
        "value": original,
        "section_key": sec,
        "STATE_CODE": state,
        "SURVEY_YEAR": sy,
    })

    counts = {
        "tier0_observed": int(original.notna().sum()),
        "tier1_section": 0,
        "tier2_state_year": 0,
        "tier3_state": 0,
        "tier4_global": 0,
        "tier5_zero": 0,
    }

    def apply_fill(fill_values: pd.Series, tier_id: int, counter: str) -> None:
        nonlocal dense, tier
        mask = dense.isna() & fill_values.notna()
        if mask.any():
            dense.loc[mask] = fill_values.loc[mask]
            tier.loc[mask] = tier_id
            counts[counter] = int(mask.sum())

    sec_med = work.groupby("section_key")["value"].transform("median")
    apply_fill(sec_med, 1, "tier1_section")

    state_year_med = work.groupby(["STATE_CODE", "SURVEY_YEAR"])["value"].transform("median")
    apply_fill(state_year_med, 2, "tier2_state_year")

    state_med = work.groupby("STATE_CODE")["value"].transform("median")
    apply_fill(state_med, 3, "tier3_state")

    global_med = original.median()
    if pd.notna(global_med):
        mask = dense.isna()
        counts["tier4_global"] = int(mask.sum())
        dense.loc[mask] = global_med
        tier.loc[mask] = 4
    else:
        mask = dense.isna()
        counts["tier5_zero"] = int(mask.sum())
        dense.loc[mask] = 0.0
        tier.loc[mask] = 5

    return dense.astype(float), tier.astype("int16"), counts


def main() -> None:
    print(f"reading {IN_CSV.name} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["SURVEY_DATE", "ORIGIN"], low_memory=False)
    source_features = [l.strip() for l in IN_FEATURES.read_text().splitlines() if l.strip()]
    sdr_features = [
        c for c in source_features
        if c in df.columns and c.startswith(SDR_PREFIXES) and not c.startswith(("STRICT_", "HAS_", "TIER_"))
    ]
    base_features = [c for c in source_features if c not in sdr_features]
    print(f"dense SDR features: {len(sdr_features)}")

    coverage_rows = []
    reliability_features = []

    for feature in sdr_features:
        strict = finite_numeric(df[feature])
        df[f"STRICT_{feature}"] = strict
        df[f"HAS_{feature}"] = strict.notna().astype("int8")

        dense, tier, counts = densify_feature(df, feature)
        df[feature] = dense
        df[f"TIER_{feature}"] = tier
        reliability_features.extend([f"HAS_{feature}", f"TIER_{feature}"])

        row = {
            "feature": feature,
            "strict_non_null": int(strict.notna().sum()),
            "strict_coverage": float(strict.notna().mean()),
            "dense_non_null": int(dense.notna().sum()),
            "dense_coverage": float(dense.notna().mean()),
        }
        row.update(counts)
        coverage_rows.append(row)

    features = []
    for feature in base_features + sdr_features + reliability_features:
        if feature in df.columns and feature not in features:
            features.append(feature)
    OUT_FEATURES.write_text("\n".join(features) + "\n")

    df.to_csv(OUT_CSV, index=False)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
    except Exception as exc:
        print(f"parquet skipped: {exc}")

    cov = pd.DataFrame(coverage_rows)
    cov.to_csv(OUTDIR / "coverage_dense.csv", index=False)
    if not cov.empty:
        cov.sort_values("strict_coverage").head(25).to_csv(OUTDIR / "sparsest_features.csv", index=False)

    lines = [
        "# Aggressive Dense SDR Dataset",
        "",
        f"Source tag: `{SOURCE_TAG}`",
        f"Output tag: `{OUT_TAG}`",
        f"Rows: {len(df)}",
        f"Densified SDR features: {len(sdr_features)}",
        "",
        "Strict observed values are preserved as `STRICT_<feature>` columns.",
        "Model-facing SDR columns are filled by section, state-year, state, then global medians.",
        "`HAS_<feature>` and `TIER_<feature>` columns document measured versus borrowed values.",
    ]
    OUT_DICT.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_FEATURES}")
    print(f"coverage -> {OUTDIR / 'coverage_dense.csv'}")


if __name__ == "__main__":
    main()
