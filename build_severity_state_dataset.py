"""
Build a dataset with LTPP-faithful disjoint fatigue severity-state targets.

The existing multi-head target uses cumulative severity-or-greater bands:
  C_PCT_L = low + moderate + high
  C_PCT_M = moderate + high
  C_PCT_H = high

This script keeps those columns for comparison, but adds raw LTPP severity states as disjoint
percent lane-area targets:
  C_PCT_LOW_ONLY
  C_PCT_MOD_ONLY
  C_PCT_HIGH_ONLY

The raw SDR fields are GATOR_CRACK_A_L/M/H, where LTPP defines severity visually:
  low: no/few connecting cracks, no spalling/sealing/pumping
  moderate: interconnected cracks, possible slight spalling/sealing/pumping
  high: moderately/severely spalled interconnected cracks, may be sealed/pumping

Run:
  python3 build_severity_state_dataset.py

Optional:
  STATE_SOURCE_TAG=_sdr_dense STATE_OUT_TAG=_sdr_dense_state python3 build_severity_state_dataset.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import ltpp_io as io


PROJ = Path(__file__).resolve().parent
SOURCE_TAG = os.environ.get("STATE_SOURCE_TAG", "_sdr_dense")
OUT_TAG = os.environ.get("STATE_OUT_TAG", f"{SOURCE_TAG}_state")
SEC = ["STATE_CODE", "SHRP_ID"]


def severity_states() -> pd.DataFrame:
    dis = io.load_cache("ANALYSIS_DIS_AC")
    dis = io.keyize(dis)
    dis["SURVEY_DATE"] = pd.to_datetime(dis["SURVEY_DATE"], errors="coerce")
    num = lambda c: pd.to_numeric(dis[c], errors="coerce")
    A_L, A_M, A_H = num("GATOR_CRACK_A_L"), num("GATOR_CRACK_A_M"), num("GATOR_CRACK_A_H")
    width = num("SURVEY_WIDTH")
    lane_area = (width * 152.4).where(width > 0)
    measured = A_L.notna() | A_M.notna() | A_H.notna()
    aL, aM, aH = A_L.fillna(0.0), A_M.fillna(0.0), A_H.fillna(0.0)

    out = dis[SEC + ["SURVEY_DATE", "CONSTRUCTION_NO", "SURVEY_WIDTH", "GATOR_CRACK_A_FLAG"]].copy()
    out["GATOR_AREA_LOW_SQM"] = aL.where(measured)
    out["GATOR_AREA_MOD_SQM"] = aM.where(measured)
    out["GATOR_AREA_HIGH_SQM"] = aH.where(measured)
    out["C_PCT_LOW_ONLY"] = (100.0 * aL / lane_area).where(measured).clip(0, 100)
    out["C_PCT_MOD_ONLY"] = (100.0 * aM / lane_area).where(measured).clip(0, 100)
    out["C_PCT_HIGH_ONLY"] = (100.0 * aH / lane_area).where(measured).clip(0, 100)
    out["C_PCT_TOTAL_STATE"] = (out["C_PCT_LOW_ONLY"] + out["C_PCT_MOD_ONLY"] + out["C_PCT_HIGH_ONLY"]).clip(0, 100)
    out["C_PCT_MOD_OR_HIGH_STATE"] = (out["C_PCT_MOD_ONLY"] + out["C_PCT_HIGH_ONLY"]).clip(0, 100)
    out["SEVERITY_AREA_INDEX"] = (
        out["C_PCT_LOW_ONLY"] + 2.0 * out["C_PCT_MOD_ONLY"] + 3.0 * out["C_PCT_HIGH_ONLY"]
    )
    out["SEVERITY_MEAN_CLASS"] = np.divide(
        out["SEVERITY_AREA_INDEX"],
        out["C_PCT_TOTAL_STATE"],
        out=np.zeros(len(out), dtype=float),
        where=out["C_PCT_TOTAL_STATE"].to_numpy(float) > 0,
    )
    out["GATOR_STATE_FLAG"] = out["GATOR_CRACK_A_FLAG"]
    keep = SEC + [
        "SURVEY_DATE",
        "CONSTRUCTION_NO",
        "SURVEY_WIDTH",
        "GATOR_AREA_LOW_SQM",
        "GATOR_AREA_MOD_SQM",
        "GATOR_AREA_HIGH_SQM",
        "C_PCT_LOW_ONLY",
        "C_PCT_MOD_ONLY",
        "C_PCT_HIGH_ONLY",
        "C_PCT_TOTAL_STATE",
        "C_PCT_MOD_OR_HIGH_STATE",
        "SEVERITY_AREA_INDEX",
        "SEVERITY_MEAN_CLASS",
        "GATOR_STATE_FLAG",
    ]
    out = out[keep].dropna(subset=["SURVEY_DATE"])
    out = out.sort_values("C_PCT_TOTAL_STATE").drop_duplicates(SEC + ["SURVEY_DATE"], keep="last")
    return out


def main() -> None:
    src = PROJ / f"fatigue_svecd{SOURCE_TAG}.csv"
    df = io.keyize(pd.read_csv(src, parse_dates=["SURVEY_DATE", "ORIGIN"], low_memory=False))
    sev = severity_states()
    n0 = len(df)
    drop_existing = [c for c in sev.columns if c in df.columns and c not in SEC + ["SURVEY_DATE"]]
    if drop_existing:
        df = df.drop(columns=drop_existing)
    merged = df.merge(sev, on=SEC + ["SURVEY_DATE"], how="left")
    if len(merged) != n0:
        raise RuntimeError(f"join changed row count {n0}->{len(merged)}")

    state_cols = ["C_PCT_LOW_ONLY", "C_PCT_MOD_ONLY", "C_PCT_HIGH_ONLY"]
    have = merged[state_cols].notna().any(axis=1)
    # Keep old cumulative targets synchronized with raw state fields for this state dataset.
    merged["C_PCT_L"] = merged["C_PCT_TOTAL_STATE"]
    merged["C_PCT_M"] = merged["C_PCT_MOD_OR_HIGH_STATE"]
    merged["C_PCT_H"] = merged["C_PCT_HIGH_ONLY"]

    out_csv = PROJ / f"fatigue_svecd{OUT_TAG}.csv"
    out_parquet = PROJ / f"fatigue_svecd{OUT_TAG}.parquet"
    merged.to_csv(out_csv, index=False)
    try:
        merged.to_parquet(out_parquet, index=False)
    except Exception as exc:
        print(f"parquet skipped: {exc}")

    features = (PROJ / f"feature_cols{SOURCE_TAG}.txt").read_text()
    (PROJ / f"feature_cols{OUT_TAG}.txt").write_text(features)

    stats = []
    for c in state_cols + ["C_PCT_TOTAL_STATE", "C_PCT_MOD_OR_HIGH_STATE", "SEVERITY_AREA_INDEX"]:
        s = merged.loc[have, c]
        stats.append({
            "target": c,
            "non_null": int(s.notna().sum()),
            "zero_pct": float((s == 0).mean()),
            "nonzero_median": float(s[s > 0].median()) if (s > 0).any() else np.nan,
            "max": float(s.max()),
        })
    pd.DataFrame(stats).to_csv(PROJ / f"outputs_severity_state_targets{OUT_TAG}.csv", index=False)

    lines = [
        "# LTPP Severity-State Dataset",
        "",
        f"Source tag: `{SOURCE_TAG}`",
        f"Output tag: `{OUT_TAG}`",
        f"Rows: {len(merged)}",
        f"Rows with state targets: {int(have.sum())}",
        "",
        "Adds disjoint LTPP fatigue severity states:",
        "",
        "- `C_PCT_LOW_ONLY`: low-severity alligator/fatigue area percent",
        "- `C_PCT_MOD_ONLY`: moderate-severity alligator/fatigue area percent",
        "- `C_PCT_HIGH_ONLY`: high-severity alligator/fatigue area percent",
        "",
        "LTPP severity descriptions come from the SDR InfoPave data dictionary for `ANALYSIS_DIS_AC`.",
    ]
    (PROJ / f"data_dictionary{OUT_TAG}.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {out_csv}")
    print(pd.DataFrame(stats).to_string(index=False))


if __name__ == "__main__":
    main()
