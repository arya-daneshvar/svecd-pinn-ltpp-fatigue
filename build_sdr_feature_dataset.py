"""
Build an SDR-augmented fatigue dataset for the pure PINN experiments.

The row frame and targets start from fatigue_svecd_mhB.csv, because that file already encodes
the cleaned GPS-1/SPS-1 distress progression, LEA strain, S-VECD variables, binder/relaxation
features, and cumulative severity targets.  This script adds new features from raw SDR modules
that were not part of the Claude-proposed feature set:

  * daily climate: freeze/thaw, hot days, precipitation/snow, humidity, wind
  * seasonal monitoring: pavement temperature, TDR moisture, interpreted freeze state
  * monthly traffic: class-count totals and heavy-truck shares
  * monthly GVW: gross-weight distribution summaries
  * deicing, GPR, drainage, and Dynamic Load Response static section features

Outputs:
  fatigue_svecd_sdr.csv
  fatigue_svecd_sdr.parquet (if parquet support is available)
  feature_cols_sdr.txt
  data_dictionary_sdr.md
  outputs_sdr_features/coverage.csv
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import ltpp_io as io


PROJ = Path(__file__).resolve().parent
SDR = Path(io.SDR)
CACHE = PROJ / "data_cache_sdr"
OUTDIR = PROJ / "outputs_sdr_features"
CACHE.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)

SEC = ["STATE_CODE", "SHRP_ID"]
BASE_CSV = PROJ / "fatigue_svecd_mhB.csv"
OUT_CSV = PROJ / "fatigue_svecd_sdr.csv"
OUT_PARQUET = PROJ / "fatigue_svecd_sdr.parquet"


DB = {
    "CLM_VWS_TEMP_DAILY_AL_MT": ("SDR39 - CLIMATE_DAILY_TEMP_AL_MT/SDR39 - CLIMATE_DAILY_TEMP_AL_MT\\Climate_Daily_Temp_AL_MT.accdb", "CLM_VWS_TEMP_DAILY"),
    "CLM_VWS_TEMP_DAILY_NE_SK": ("SDR39 - CLIMATE_DAILY_TEMP_NE_SK/Climate_Daily_Temp_NE_SK.accdb", "CLM_VWS_TEMP_DAILY"),
    "CLM_VWS_PRECIP_DAILY_AL_MT": ("SDR39 - CLIMATE_DAILY_PRECIP_AL_MT/SDR39 - CLIMATE_DAILY_PRECIP_AL_MT\\Climate_Daily_Precip_AL_MT.accdb", "CLM_VWS_PRECIP_DAILY"),
    "CLM_VWS_PRECIP_DAILY_NE_SK": ("SDR39 - CLIMATE_DAILY_PRECIP_NE_SK/SDR39 - CLIMATE_DAILY_PRECIP_NE_SK\\Climate_Daily_Precip_NE_SK.accdb", "CLM_VWS_PRECIP_DAILY"),
    "CLM_VWS_HUMIDITY_DAILY": ("SDR39 - CLIMATE_DAILY_HUMIDITY/Climate_Daily_Humidity.accdb", "CLM_VWS_HUMIDITY_DAILY"),
    "CLM_VWS_WIND_DAILY": ("SDR39 - CLIMATE_DAILY_WIND/Climate_Daily_Wind.accdb", "CLM_VWS_WIND_DAILY"),
    "MM_CT": ("SDR39 - MONTHLY_COUNT/SDR39 - MONTHLY_COUNT\\Monthly_Count.accdb", "MM_CT"),
    "MM_GVW_AL_CO": ("SDR39 - MONTHLY_GVW_AL_CO/SDR39 - MONTHLY_GVW_AL_CO\\Monthly_GVW_AL_CO.accdb", "MM_GVW"),
    "MM_GVW_CT_MO": ("SDR39 - MONTHLY_GVW_CT_MO/SDR39 - MONTHLY_GVW_CT_MO\\Monthly_GVW_CT_MO.accdb", "MM_GVW"),
    "MM_GVW_MT_UT": ("SDR39 - MONTHLY_GVW_MT_UT/SDR39 - MONTHLY_GVW_MT_UT\\Monthly_GVW_MT_UT.accdb", "MM_GVW"),
    "MM_GVW_VT_SK": ("SDR39 - MONTHLY_GVW_VT_SK/SDR39 - MONTHLY_GVW_VT_SK\\Monthly_GVW_VT_SK.accdb", "MM_GVW"),
    "SMP_MRCTEMP_AUTO_DAY_STATS": ("SDR39 - SEASONAL_MONITORING_MRCTEMP/SDR39 - SEASONAL_MONITORING_MRCTEMP\\Seasonal_Monitoring_MRCTemp.accdb", "SMP_MRCTEMP_AUTO_DAY_STATS"),
    "SMP_TDR_AUTO_MOISTURE": ("SDR39 - SEASONAL_MONITORING/Seasonal_Monitoring.accdb", "SMP_TDR_AUTO_MOISTURE"),
    "SMP_FREEZE_STATE": ("SDR39 - SEASONAL_MONITORING/Seasonal_Monitoring.accdb", "SMP_FREEZE_STATE"),
    "INV_DEICE_SITE_DATA": ("SDR39 - DEICING/Deicing.accdb", "INV_DEICE_SITE_DATA"),
    "GPR_THICK_SECT": ("SDR39 - GROUND_PENETRATING_RADAR/Ground_Penetrating_Radar.accdb", "GPR_THICK_SECT"),
    "ANALYSIS_DRAINAGE": ("SDR39 - DRAINAGE/Drainage.accdb", "ANALYSIS_DRAINAGE"),
    "DLR_OH_STRAIN_TRACE_SUM_AC": ("SDR39 - DYNAMIC_LOAD_RESPONSE/Dynamic_Load_Response.accdb", "DLR_OH_STRAIN_TRACE_SUM_AC"),
}


def export_table(name: str) -> Path:
    rel, table = DB[name]
    dst = CACHE / f"{name}.csv"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    src = SDR / rel
    if not src.exists():
        raise FileNotFoundError(src)
    print(f"export {name} ...", flush=True)
    with open(dst, "w") as fh:
        subprocess.run(["mdb-export", str(src), table], stdout=fh, check=True)
    return dst


def norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    return io.keyize(df)


def read_small(name: str, **kw) -> pd.DataFrame:
    return norm_keys(pd.read_csv(export_table(name), low_memory=False, **kw))


def base_df() -> pd.DataFrame:
    df = pd.read_csv(BASE_CSV, parse_dates=["SURVEY_DATE", "ORIGIN"], low_memory=False)
    df = norm_keys(df)
    df["SURVEY_YEAR"] = df["SURVEY_DATE"].dt.year.astype("Int64")
    df["ORIGIN_YEAR"] = df["ORIGIN"].dt.year.astype("Int64")
    return df


def relevant_vws(df: pd.DataFrame) -> pd.DataFrame:
    cl = io.load_cache("CLM_SITE_VWS_LINK")
    cl = cl[SEC + ["VWS_ID"]].dropna().drop_duplicates()
    cl["VWS_ID"] = cl["VWS_ID"].astype(str).str.strip().str.zfill(6)
    cl = cl.groupby(SEC, as_index=False)["VWS_ID"].first()
    return df[SEC].drop_duplicates().merge(cl, on=SEC, how="left")


def annual_daily_features(names: list[str], usecols: list[str], kind: str, vws_ids: set[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        path = export_table(name)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=400_000, low_memory=False):
            chunk["VWS_ID"] = chunk["VWS_ID"].astype(str).str.strip().str.zfill(6)
            chunk = chunk[chunk["VWS_ID"].isin(vws_ids)].copy()
            if chunk.empty:
                continue
            chunk["VWS_DATE"] = pd.to_datetime(chunk["VWS_DATE"], errors="coerce")
            chunk = chunk.dropna(subset=["VWS_DATE"])
            chunk["YEAR"] = chunk["VWS_DATE"].dt.year.astype(int)
            for c in chunk.columns:
                if c not in {"VWS_ID", "VWS_DATE", "YEAR", "RECORD_STATUS"}:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            if kind == "temp":
                chunk["CLM_FREEZE_THAW_DAYS"] = ((chunk["MIN_DAY_TEMP"] < 0) & (chunk["MAX_DAY_TEMP"] > 0)).astype(float)
                chunk["CLM_FREEZE_DAYS"] = (chunk["MIN_DAY_TEMP"] < 0).astype(float)
                chunk["CLM_HOT30_DAYS"] = (chunk["MAX_DAY_TEMP"] >= 30).astype(float)
                chunk["CLM_HOT35_DAYS"] = (chunk["MAX_DAY_TEMP"] >= 35).astype(float)
                chunk["CLM_TEMP_RANGE"] = chunk["MAX_DAY_TEMP"] - chunk["MIN_DAY_TEMP"]
                agg = chunk.groupby(["VWS_ID", "YEAR"], as_index=False).agg(
                    CLM_MEAN_TEMP=("MEAN_DAY_TEMP", "mean"),
                    CLM_MAX_TEMP_MEAN=("MAX_DAY_TEMP", "mean"),
                    CLM_MIN_TEMP_MEAN=("MIN_DAY_TEMP", "mean"),
                    CLM_TEMP_RANGE_MEAN=("CLM_TEMP_RANGE", "mean"),
                    CLM_FREEZE_THAW_DAYS=("CLM_FREEZE_THAW_DAYS", "sum"),
                    CLM_FREEZE_DAYS=("CLM_FREEZE_DAYS", "sum"),
                    CLM_HOT30_DAYS=("CLM_HOT30_DAYS", "sum"),
                    CLM_HOT35_DAYS=("CLM_HOT35_DAYS", "sum"),
                )
            elif kind == "precip":
                chunk["CLM_WET_DAYS"] = (chunk["DAY_PRECIPITATION"] > 0).astype(float)
                chunk["CLM_SNOW_DAYS"] = (chunk["DAY_SNOWFALL"] > 0).astype(float)
                agg = chunk.groupby(["VWS_ID", "YEAR"], as_index=False).agg(
                    CLM_PRECIP_SUM=("DAY_PRECIPITATION", "sum"),
                    CLM_SNOW_SUM=("DAY_SNOWFALL", "sum"),
                    CLM_WET_DAYS=("CLM_WET_DAYS", "sum"),
                    CLM_SNOW_DAYS=("CLM_SNOW_DAYS", "sum"),
                )
            elif kind == "humidity":
                agg = chunk.groupby(["VWS_ID", "YEAR"], as_index=False).agg(
                    CLM_HUM_MAX_MEAN=("MAX_DAY_HUM", "mean"),
                    CLM_HUM_MIN_MEAN=("MIN_DAY_HUM", "mean"),
                )
            else:
                agg = chunk.groupby(["VWS_ID", "YEAR"], as_index=False).agg(
                    CLM_WIND_MEAN=("MEAN_DAY_WIND_SPD", "mean"),
                    CLM_WIND_MAX_MEAN=("MAX_DAY_WIND_SPD", "mean"),
                )
            rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=["VWS_ID", "YEAR"])
    out = pd.concat(rows, ignore_index=True)
    num = [c for c in out.columns if c not in {"VWS_ID", "YEAR"}]
    return out.groupby(["VWS_ID", "YEAR"], as_index=False)[num].mean()


def add_interval_features(df: pd.DataFrame, annual: pd.DataFrame, id_col: str, prefix_cols: list[str], prefix: str) -> pd.DataFrame:
    if annual.empty:
        return df
    annual = annual.sort_values([id_col, "YEAR"]).copy()
    recent = annual.copy()
    recent["SURVEY_YEAR"] = recent["YEAR"] + 1
    recent = recent.rename(columns={c: f"{prefix}_PREVYR_{c}" for c in prefix_cols})
    df = df.merge(recent[[id_col, "SURVEY_YEAR"] + [f"{prefix}_PREVYR_{c}" for c in prefix_cols]],
                  on=[id_col, "SURVEY_YEAR"], how="left")

    # cumulative from origin calendar year through the year before survey.
    out = []
    for key, g in annual.groupby(id_col):
        gy = g.sort_values("YEAR")
        years = gy["YEAR"].to_numpy()
        vals = gy[prefix_cols].fillna(0.0).to_numpy()
        cs = np.vstack([np.zeros((1, vals.shape[1])), np.cumsum(vals, axis=0)])
        out.append((key, years, cs))

    lookup = {k: (years, cs) for k, years, cs in out}
    rows = np.full((len(df), len(prefix_cols)), np.nan)
    for i, row in enumerate(df[[id_col, "ORIGIN_YEAR", "SURVEY_YEAR"]].itertuples(index=False, name=None)):
        key, oy, sy = row
        if pd.isna(key) or pd.isna(oy) or pd.isna(sy) or key not in lookup:
            continue
        years, cs = lookup[key]
        lo = np.searchsorted(years, int(oy), side="left")
        hi = np.searchsorted(years, int(sy), side="left")
        if hi > lo:
            rows[i] = cs[hi] - cs[lo]
    for j, c in enumerate(prefix_cols):
        df[f"{prefix}_CUM_{c}"] = rows[:, j]
    return df


def add_daily_climate(df: pd.DataFrame) -> pd.DataFrame:
    sec_vws = relevant_vws(df)
    df = df.merge(sec_vws, on=SEC, how="left")
    vws_ids = set(sec_vws["VWS_ID"].dropna().astype(str))
    print(f"daily climate VWS coverage: {len(vws_ids)} stations", flush=True)

    specs = [
        (["CLM_VWS_TEMP_DAILY_AL_MT", "CLM_VWS_TEMP_DAILY_NE_SK"],
         ["VWS_ID", "VWS_DATE", "MEAN_DAY_TEMP", "MAX_DAY_TEMP", "MIN_DAY_TEMP"], "temp"),
        (["CLM_VWS_PRECIP_DAILY_AL_MT", "CLM_VWS_PRECIP_DAILY_NE_SK"],
         ["VWS_ID", "VWS_DATE", "DAY_PRECIPITATION", "DAY_SNOWFALL"], "precip"),
        (["CLM_VWS_HUMIDITY_DAILY"], ["VWS_ID", "VWS_DATE", "MAX_DAY_HUM", "MIN_DAY_HUM"], "humidity"),
        (["CLM_VWS_WIND_DAILY"], ["VWS_ID", "VWS_DATE", "MEAN_DAY_WIND_SPD", "MAX_DAY_WIND_SPD"], "wind"),
    ]
    for names, cols, kind in specs:
        annual = annual_daily_features(names, cols, kind, vws_ids)
        num = [c for c in annual.columns if c not in {"VWS_ID", "YEAR"}]
        df = add_interval_features(df, annual, "VWS_ID", num, f"CLM_{kind.upper()}")
    return df


def add_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    # MRCTemp daily stats: pavement temperature at instrumented sections.
    mr = read_small("SMP_MRCTEMP_AUTO_DAY_STATS", usecols=["STATE_CODE", "SHRP_ID", "SMP_DATE", "THERM_NO", "AVG_DAY_TEMPERATURE", "MAX_TEMPERATURE", "MIN_TEMPERATURE"])
    mr["SMP_DATE"] = pd.to_datetime(mr["SMP_DATE"], errors="coerce")
    mr["YEAR"] = mr["SMP_DATE"].dt.year
    for c in ["AVG_DAY_TEMPERATURE", "MAX_TEMPERATURE", "MIN_TEMPERATURE"]:
        mr[c] = pd.to_numeric(mr[c], errors="coerce")
    mr["SMP_TEMP_RANGE"] = mr["MAX_TEMPERATURE"] - mr["MIN_TEMPERATURE"]
    ann = mr.groupby(SEC + ["YEAR"], as_index=False).agg(
        SMP_PAVTEMP_MEAN=("AVG_DAY_TEMPERATURE", "mean"),
        SMP_PAVTEMP_MAX_MEAN=("MAX_TEMPERATURE", "mean"),
        SMP_PAVTEMP_MIN_MEAN=("MIN_TEMPERATURE", "mean"),
        SMP_PAVTEMP_RANGE_MEAN=("SMP_TEMP_RANGE", "mean"),
    ).dropna(subset=["YEAR"])
    ann["SECTION_ID"] = ann["STATE_CODE"].astype(str) + "_" + ann["SHRP_ID"].astype(str)
    df["SECTION_ID"] = df["STATE_CODE"].astype(str) + "_" + df["SHRP_ID"].astype(str)
    df = add_interval_features(df, ann, "SECTION_ID",
                               ["SMP_PAVTEMP_MEAN", "SMP_PAVTEMP_MAX_MEAN", "SMP_PAVTEMP_MIN_MEAN", "SMP_PAVTEMP_RANGE_MEAN"],
                               "SMP_TEMP")

    tdr = read_small("SMP_TDR_AUTO_MOISTURE", usecols=["STATE_CODE", "SHRP_ID", "SMP_DATE", "VOLUMETRIC_MOISTURE_CONTENT", "GRAVIMETRIC_MOISTURE_CONTENT"])
    tdr["SMP_DATE"] = pd.to_datetime(tdr["SMP_DATE"], errors="coerce")
    tdr["YEAR"] = tdr["SMP_DATE"].dt.year
    for c in ["VOLUMETRIC_MOISTURE_CONTENT", "GRAVIMETRIC_MOISTURE_CONTENT"]:
        tdr[c] = pd.to_numeric(tdr[c], errors="coerce")
    ann = tdr.groupby(SEC + ["YEAR"], as_index=False).agg(
        SMP_TDR_VOL_MOIST_MEAN=("VOLUMETRIC_MOISTURE_CONTENT", "mean"),
        SMP_TDR_VOL_MOIST_MAX=("VOLUMETRIC_MOISTURE_CONTENT", "max"),
        SMP_TDR_GRAV_MOIST_MEAN=("GRAVIMETRIC_MOISTURE_CONTENT", "mean"),
    ).dropna(subset=["YEAR"])
    ann["SECTION_ID"] = ann["STATE_CODE"].astype(str) + "_" + ann["SHRP_ID"].astype(str)
    df = add_interval_features(df, ann, "SECTION_ID",
                               ["SMP_TDR_VOL_MOIST_MEAN", "SMP_TDR_VOL_MOIST_MAX", "SMP_TDR_GRAV_MOIST_MEAN"],
                               "SMP_MOIST")

    fr = read_small("SMP_FREEZE_STATE", usecols=["STATE_CODE", "SHRP_ID", "SMP_DATE", "INTERPRET_DEPTH", "FREEZE_STATE", "INTERPRET_SOIL_TEMP"])
    fr["SMP_DATE"] = pd.to_datetime(fr["SMP_DATE"], errors="coerce")
    fr["YEAR"] = fr["SMP_DATE"].dt.year
    fr["INTERPRET_DEPTH"] = pd.to_numeric(fr["INTERPRET_DEPTH"], errors="coerce")
    fr["INTERPRET_SOIL_TEMP"] = pd.to_numeric(fr["INTERPRET_SOIL_TEMP"], errors="coerce")
    fr["IS_FROZEN"] = fr["FREEZE_STATE"].astype(str).str.upper().eq("F").astype(float)
    ann = fr.groupby(SEC + ["YEAR"], as_index=False).agg(
        SMP_FREEZE_OBS=("IS_FROZEN", "sum"),
        SMP_FREEZE_DEPTH_MAX=("INTERPRET_DEPTH", "max"),
        SMP_SOIL_TEMP_MEAN=("INTERPRET_SOIL_TEMP", "mean"),
    ).dropna(subset=["YEAR"])
    ann["SECTION_ID"] = ann["STATE_CODE"].astype(str) + "_" + ann["SHRP_ID"].astype(str)
    df = add_interval_features(df, ann, "SECTION_ID",
                               ["SMP_FREEZE_OBS", "SMP_FREEZE_DEPTH_MAX", "SMP_SOIL_TEMP_MEAN"],
                               "SMP_FREEZE")
    return df


def annual_section_features(df_raw: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    df_raw = norm_keys(df_raw)
    df_raw["YEAR"] = pd.to_numeric(df_raw["YEAR"], errors="coerce")
    return df_raw.groupby(SEC + ["YEAR"], as_index=False)[value_cols].sum().dropna(subset=["YEAR"])


def add_monthly_count(df: pd.DataFrame) -> pd.DataFrame:
    path = export_table("MM_CT")
    rows = []
    usecols = ["STATE_CODE", "SHRP_ID", "YEAR", "MONTH", "NUM_DAY_OCCURRENCES"] + [f"CT_SUM_{i:02d}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]]
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=300_000, low_memory=False):
        chunk = norm_keys(chunk)
        for c in chunk.columns:
            if c not in SEC:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        occ = chunk.get("NUM_DAY_OCCURRENCES", 1).fillna(1)
        cols = [c for c in chunk.columns if c.startswith("CT_SUM_")]
        heavy = [c for c in cols if c not in {"CT_SUM_01", "CT_SUM_02", "CT_SUM_03"}]
        chunk["TRF_MONTHLY_CT_TOTAL"] = chunk[cols].fillna(0).sum(axis=1) * occ
        chunk["TRF_MONTHLY_CT_HEAVY"] = chunk[heavy].fillna(0).sum(axis=1) * occ
        rows.append(chunk[SEC + ["YEAR", "TRF_MONTHLY_CT_TOTAL", "TRF_MONTHLY_CT_HEAVY"]])
    ann = annual_section_features(pd.concat(rows, ignore_index=True), ["TRF_MONTHLY_CT_TOTAL", "TRF_MONTHLY_CT_HEAVY"])
    ann["TRF_MONTHLY_HEAVY_SHARE"] = ann["TRF_MONTHLY_CT_HEAVY"] / ann["TRF_MONTHLY_CT_TOTAL"].replace(0, np.nan)
    ann["SECTION_ID"] = ann["STATE_CODE"].astype(str) + "_" + ann["SHRP_ID"].astype(str)
    df["SECTION_ID"] = df["STATE_CODE"].astype(str) + "_" + df["SHRP_ID"].astype(str)
    return add_interval_features(df, ann, "SECTION_ID",
                                 ["TRF_MONTHLY_CT_TOTAL", "TRF_MONTHLY_CT_HEAVY", "TRF_MONTHLY_HEAVY_SHARE"],
                                 "TRF_MC")


def add_monthly_gvw(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bin_cols = [f"BIN{i:02d}" for i in range(1, 51)]
    for name in ["MM_GVW_AL_CO", "MM_GVW_CT_MO", "MM_GVW_MT_UT", "MM_GVW_VT_SK"]:
        path = export_table(name)
        for chunk in pd.read_csv(path, usecols=lambda c: c in set(SEC + ["YEAR", "MONTH", "NUM_DAY_OCCURRENCES"] + bin_cols), chunksize=250_000, low_memory=False):
            chunk = norm_keys(chunk)
            for c in chunk.columns:
                if c not in SEC:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            present = [c for c in bin_cols if c in chunk.columns]
            vals = chunk[present].fillna(0.0).to_numpy()
            idx = np.array([int(c[-2:]) for c in present], dtype=float)
            tot = vals.sum(axis=1)
            high = vals[:, idx >= 30].sum(axis=1) if len(idx) else np.zeros(len(chunk))
            mean_idx = np.divide(vals @ idx, tot, out=np.full(len(chunk), np.nan), where=tot > 0)
            chunk["TRF_GVW_OBS"] = tot
            chunk["TRF_GVW_MEAN_BIN"] = mean_idx
            chunk["TRF_GVW_HIGHBIN_SHARE"] = np.divide(high, tot, out=np.full(len(chunk), np.nan), where=tot > 0)
            rows.append(chunk[SEC + ["YEAR", "TRF_GVW_OBS", "TRF_GVW_MEAN_BIN", "TRF_GVW_HIGHBIN_SHARE"]])
    if not rows:
        return df
    raw = pd.concat(rows, ignore_index=True)
    ann = raw.groupby(SEC + ["YEAR"], as_index=False).agg(
        TRF_GVW_OBS=("TRF_GVW_OBS", "sum"),
        TRF_GVW_MEAN_BIN=("TRF_GVW_MEAN_BIN", "mean"),
        TRF_GVW_HIGHBIN_SHARE=("TRF_GVW_HIGHBIN_SHARE", "mean"),
    ).dropna(subset=["YEAR"])
    ann["SECTION_ID"] = ann["STATE_CODE"].astype(str) + "_" + ann["SHRP_ID"].astype(str)
    df["SECTION_ID"] = df["STATE_CODE"].astype(str) + "_" + df["SHRP_ID"].astype(str)
    return add_interval_features(df, ann, "SECTION_ID",
                                 ["TRF_GVW_OBS", "TRF_GVW_MEAN_BIN", "TRF_GVW_HIGHBIN_SHARE"],
                                 "TRF_GVW")


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    deice = read_small("INV_DEICE_SITE_DATA")
    for c in ["FREQ_SNOW_REMOVAL", "FREQ_DEICE_APPL"]:
        deice[c] = pd.to_numeric(deice[c], errors="coerce")
    deice = deice.groupby(SEC, as_index=False).agg(
        DEICE_SNOW_REMOVAL_FREQ=("FREQ_SNOW_REMOVAL", "median"),
        DEICE_APPL_FREQ=("FREQ_DEICE_APPL", "median"),
    )
    df = df.merge(deice, on=SEC, how="left")

    gpr = read_small("GPR_THICK_SECT")
    for c in ["LAYER_THICK_MEAN", "LAYER_THICK_STD", "LAYER_DIELECTRIC_MEAN", "LAYER_DIELECTRIC_STD"]:
        gpr[c] = pd.to_numeric(gpr[c], errors="coerce")
    ac = gpr[gpr["LAYER_TYPE_SECTION"].astype(str).str.upper().eq("AC")].copy()
    ac = ac.groupby(SEC + ["GPR_DATE", "LANE_POSITION"], as_index=False).agg(
        GPR_AC_THICK_MEAN_SUM=("LAYER_THICK_MEAN", "sum"),
        GPR_AC_THICK_STD_MEAN=("LAYER_THICK_STD", "mean"),
        GPR_AC_DIEL_MEAN=("LAYER_DIELECTRIC_MEAN", "mean"),
        GPR_AC_DIEL_STD_MEAN=("LAYER_DIELECTRIC_STD", "mean"),
    )
    ac = ac.groupby(SEC, as_index=False).median(numeric_only=True)
    df = df.merge(ac, on=SEC, how="left")

    dr = read_small("ANALYSIS_DRAINAGE")
    for c in ["HYDRAULIC_CONDUCTIVITY", "CROSS_SLOPE", "GRADE", "DRAINPIPE_DIAMETER", "LATERALS_SPACING", "PIPE_DEPTH", "PIPE_DIAMETER"]:
        if c in dr.columns:
            dr[c] = pd.to_numeric(dr[c], errors="coerce")
    dr["DRAINAGE_PRESENT"] = dr["DRAINAGE_LOCATION"].notna().astype(float)
    keep = [c for c in ["HYDRAULIC_CONDUCTIVITY", "CROSS_SLOPE", "GRADE", "PIPE_DEPTH", "PIPE_DIAMETER", "DRAINAGE_PRESENT"] if c in dr.columns]
    dr = dr.groupby(SEC, as_index=False)[keep].median(numeric_only=True)
    dr = dr.rename(columns={c: f"DRAIN_{c}" for c in keep})
    df = df.merge(dr, on=SEC, how="left")

    dlr = read_small("DLR_OH_STRAIN_TRACE_SUM_AC")
    strain_cols = [c for c in dlr.columns if c.startswith("STRAIN_VALUE_SMOOTH_")]
    for c in strain_cols:
        dlr[c] = pd.to_numeric(dlr[c], errors="coerce")
    if strain_cols:
        arr = dlr[strain_cols].to_numpy(float)
        dlr["DLR_STRAIN_ABS_MAX"] = np.nanmax(np.abs(arr), axis=1)
        dlr["DLR_STRAIN_POS_MAX"] = np.nanmax(arr, axis=1)
        dlr["DLR_STRAIN_NEG_MIN"] = np.nanmin(arr, axis=1)
        dlr = dlr.groupby(SEC, as_index=False).agg(
            DLR_STRAIN_ABS_MAX=("DLR_STRAIN_ABS_MAX", "max"),
            DLR_STRAIN_POS_MAX=("DLR_STRAIN_POS_MAX", "max"),
            DLR_STRAIN_NEG_MIN=("DLR_STRAIN_NEG_MIN", "min"),
        )
        df = df.merge(dlr, on=SEC, how="left")
    return df


def write_feature_list(df: pd.DataFrame):
    base = [l.strip() for l in (PROJ / "feature_cols_mhB.txt").read_text().splitlines() if l.strip()]
    added = [c for c in df.columns if c.startswith(("CLM_", "SMP_", "TRF_MC_", "TRF_GVW_", "DEICE_", "GPR_", "DRAIN_", "DLR_"))]
    # Do not include raw helper ids as features.
    features = []
    for c in base + added:
        if c in df.columns and c not in features:
            features.append(c)
    (PROJ / "feature_cols_sdr.txt").write_text("\n".join(features) + "\n")
    cov = []
    for c in added:
        cov.append({"feature": c, "non_null": int(df[c].notna().sum()), "coverage": float(df[c].notna().mean())})
    pd.DataFrame(cov).sort_values("coverage", ascending=False).to_csv(OUTDIR / "coverage.csv", index=False)


def write_dictionary(df: pd.DataFrame):
    lines = [
        "# SDR-Augmented Dataset",
        "",
        f"Rows: {len(df)}",
        f"Sections: {df[SEC].drop_duplicates().shape[0]}",
        "",
        "Adds features from daily climate, seasonal monitoring, monthly traffic/count/GVW, deicing, GPR, drainage, and dynamic load response.",
        "Base rows/targets/mechanistic columns come from `fatigue_svecd_mhB.csv`.",
    ]
    (PROJ / "data_dictionary_sdr.md").write_text("\n".join(lines) + "\n")


def main():
    print("loading base fatigue_svecd_mhB ...")
    df = base_df()
    print(f"base rows={len(df)} sections={df[SEC].drop_duplicates().shape[0]}")
    df = add_daily_climate(df)
    df = add_seasonal(df)
    df = add_monthly_count(df)
    df = add_monthly_gvw(df)
    df = add_static_features(df)
    # Keep SECTION_ID because train_models.py uses it for section grouping and temporal splits.
    if "SECTION_ID" not in df.columns:
        df["SECTION_ID"] = df["STATE_CODE"].astype(str) + "_" + df["SHRP_ID"].astype(str)
    df.to_csv(OUT_CSV, index=False)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
    except Exception as exc:
        print(f"parquet skipped: {exc}")
    write_feature_list(df)
    write_dictionary(df)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {PROJ / 'feature_cols_sdr.txt'}")
    print(f"coverage -> {OUTDIR / 'coverage.csv'}")


if __name__ == "__main__":
    main()
