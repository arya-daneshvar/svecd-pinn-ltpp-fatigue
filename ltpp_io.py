"""
ltpp_io.py
==========
Shared I/O utilities for the Claude-Third (LEA + S-VECD) fatigue pipeline.

Responsibilities
  * Locate the SDR39 databases / backcalc CSV exports.
  * Idempotently export Access tables to a local CSV cache via `mdb-export`.
  * The SHRP_ID string/int join gotcha (`norm_shrp` / `keyize`) — reused from Claude-Second.
  * Phase-0 deliverable: a *corrected* backcalculated-modulus loader that recovers per-layer
    AC / base / subgrade moduli despite the `BC_LAYER_NO` column being rendered as `True`
    in the SDR CSV export.

The fix: within each (STATE_CODE, SHRP_ID, FWD_PASS) the modulus rows are ordered top-down
(BC_LAYER_NO ascending), so row 1 = surface (AC), row 2 = base, last row = subgrade.  This is
verified against BAKCAL_LAYER_LINK + SECTION_LAYER_STRUCTURE.LAYER_TYPE (which carry a clean
BC_LAYER_NO) and against the physical expectation E_AC >> E_base, E_subgrade.

Run as a script to self-test Phase 0:
    python3 "04 - Claude/Claude - Third/ltpp_io.py"
"""

import os
import subprocess
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BASE = "/Users/arya/Documents/Dr. Golroo/009 - Thesis/01 - FATIGUE"
SDR = os.path.join(BASE, "01 - LTPP", "SDR")
# OUTDIR is the folder containing this file, so the pipeline is self-contained and works in any
# copy (Claude - Third / Fourth / Fifth ...).  Raw SDR data stays shared at BASE.
OUTDIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUTDIR, "data_cache")
BAKCAL_CSV = os.path.join(SDR, "SDR39 - BACKCALCULATION", "export_csv")
os.makedirs(CACHE, exist_ok=True)

SEC = ["STATE_CODE", "SHRP_ID"]

# (logical name) -> (database file relative to SDR, table name)
ACCDB = {
    "EXPERIMENT_SECTION":      ("SDR39 - MONITORING/Monitoring.accdb", "EXPERIMENT_SECTION"),
    "ANALYSIS_DIS_AC":         ("SDR39 - MONITORING/Monitoring.accdb", "ANALYSIS_DIS_AC"),
    "MON_DEFL_TEMP_VALUES":    ("SDR39 - MONITORING/Monitoring.accdb", "MON_DEFL_TEMP_VALUES"),
    "INV_AGE":                 ("SDR39 - INVENTORY/Inventory.accdb", "INV_AGE"),
    "INV_GRADATION":           ("SDR39 - INVENTORY/Inventory.accdb", "INV_GRADATION"),
    "INV_PMA_ASPHALT":         ("SDR39 - INVENTORY/Inventory.accdb", "INV_PMA_ASPHALT"),  # binder grade/visc/pen
    "TRF_TREND":               ("SDR39 - TRAFFIC/SDR39 - Traffic.accdb", "TRF_TREND"),
    "TRF_ESAL_INPUTS_SUMMARY": ("SDR39 - TRAFFIC/SDR39 - Traffic.accdb", "TRF_ESAL_INPUTS_SUMMARY"),
    "TST_L05B":                ("SDR39 - MATERIAL_TEST/Material_Test.accdb", "TST_L05B"),
    "ANALYSIS_TST_AC":         ("SDR39 - MATERIAL_TEST/Material_Test.accdb", "ANALYSIS_TST_AC"),
    "ANALYSIS_TST_AC_ESTAR":   ("SDR39 - MATERIAL_TEST/Material_Test.accdb", "ANALYSIS_TST_AC_ESTAR"),
    "TST_ESTAR_VOLUM_INPUT":   ("SDR39 - MATERIAL_TEST/Material_Test.accdb", "TST_ESTAR_VOLUM_INPUT"),
    "TST_ESTAR_VISC_INPUT":    ("SDR39 - MATERIAL_TEST/Material_Test.accdb", "TST_ESTAR_VISC_INPUT"),
    "CLM_VWS_TEMP_MONTH":      ("SDR39 - CLIMATE_SUMMARY_DATA/Climate_Summary_Data.accdb", "CLM_VWS_TEMP_MONTH"),
    "CLM_SITE_VWS_LINK":       ("SDR39 - CLIMATE_SUMMARY_DATA/Climate_Summary_Data.accdb", "CLM_SITE_VWS_LINK"),
    "SHRP_INFO":               ("SDR39 - MEPDG_INPUTS/SDR39 - MEPDG_INPUTS.accdb", "SHRP_INFO"),
    # NALS axle-application counts (user-added) — replaces the ESAL proxy
    "VEHICLE_CLASS_TOTAL_AXLES": ("SDR39 - NALS_AXLE_VEHICLE_COUNTS/SDR39 - NALS_AXLE_VEHICLE_COUNTS\\NALS_Axle_Vehicle_Counts.accdb", "VEHICLE_CLASS_TOTAL_AXLES"),
    "VEHICLE_CLASS_AVG_AX_ANL":  ("SDR39 - NALS_AXLE_VEHICLE_COUNTS/SDR39 - NALS_AXLE_VEHICLE_COUNTS\\NALS_Axle_Vehicle_Counts.accdb", "VEHICLE_CLASS_AVG_AX_ANL"),
}

# Unit conversions
KSI_TO_MPA = 6.894757
PSI_TO_MPA = 6.894757e-3
IN_TO_MM = 25.4


# --------------------------------------------------------------------------------------
# Extraction + key normalization
# --------------------------------------------------------------------------------------
def extract_table(name, force=False):
    """Export ACCDB[name] to data_cache/<name>.csv via mdb-export (idempotent)."""
    if name not in ACCDB:
        raise KeyError(f"{name} not in ACCDB registry")
    dst = os.path.join(CACHE, f"{name}.csv")
    if os.path.exists(dst) and os.path.getsize(dst) > 0 and not force:
        return dst
    db, tbl = ACCDB[name]
    src = os.path.join(SDR, db)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    print(f"  exporting {name} from {os.path.basename(db)} ...")
    with open(dst, "w") as fh:
        subprocess.run(["mdb-export", src, tbl], stdout=fh, check=True)
    return dst


def norm_shrp(s):
    """Normalize SHRP_ID to a zero-padded 4-char string (handles ints, quotes, floats)."""
    if pd.isna(s):
        return None
    s = str(s).strip().strip("'").strip('"')   # handle both quote styles (manual CSV splits)
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(4)


def keyize(df):
    df = df.copy()
    df["STATE_CODE"] = pd.to_numeric(df["STATE_CODE"], errors="coerce").astype("Int64")
    df["SHRP_ID"] = df["SHRP_ID"].map(norm_shrp)
    return df


def exp_label(gps_sps, experiment_no):
    """Map (GPS_SPS, EXPERIMENT_NO) -> our experiment label, else None.
    GPS-1/SPS-1: new flexible (unbound base).  GPS-2: AC on bound base.  SPS-9: SuperPave new flexible.
    All share the bottom-up fatigue mechanism."""
    g = str(gps_sps); n = str(experiment_no)
    if g == "G" and n == "1":
        return "GPS1"
    if g == "S" and n == "1":
        return "SPS1"
    if g == "G" and n == "2":
        return "GPS2"
    if g == "S" and n.startswith("9"):
        return "SPS9"
    return None


def load_cache(name, **kw):
    """Load a cached Access table (extracting it first if needed) with keys normalized."""
    extract_table(name)
    return keyize(pd.read_csv(os.path.join(CACHE, f"{name}.csv"), low_memory=False, **kw))


def load_bakcal_csv(fname, **kw):
    return keyize(pd.read_csv(os.path.join(BAKCAL_CSV, fname), low_memory=False, **kw))


# --------------------------------------------------------------------------------------
# Layer structure (clean: SECTION_LAYER_STRUCTURE has LAYER_TYPE + REPR_THICKNESS in inches)
# --------------------------------------------------------------------------------------
AC_TYPES = {"AC"}
BASE_TYPES = {"GB", "GS", "TB", "TS"}        # granular / treated base & subbase
SUBGRADE_TYPES = {"SS"}                       # subgrade soil


def load_layer_structure():
    """
    Return per (STATE_CODE, SHRP_ID, CONSTRUCTION_NO) flexible-pavement layer profile (mm):
      AC_TH_MM        — total AC thickness (sum of AC sublayers)
      BASE_TH_MM      — total base/subbase thickness (granular + treated)
      N_AC, N_BASE    — layer counts (diagnostics)
      HAS_BOUND_BASE  — any treated (TB/TS) layer present (bound base => LEA caveat)
    Subgrade is treated as a semi-infinite half-space (no thickness).
    """
    s = load_bakcal_csv("SECTION_LAYER_STRUCTURE.csv")
    s["REPR_THICKNESS"] = pd.to_numeric(s["REPR_THICKNESS"], errors="coerce")
    s["CONSTRUCTION_NO"] = pd.to_numeric(s["CONSTRUCTION_NO"], errors="coerce").astype("Int64")
    keys = SEC + ["CONSTRUCTION_NO"]

    def agg(g):
        ac = g[g["LAYER_TYPE"].isin(AC_TYPES)]
        base = g[g["LAYER_TYPE"].isin(BASE_TYPES)]
        return pd.Series({
            "AC_TH_MM": ac["REPR_THICKNESS"].sum() * IN_TO_MM,
            "BASE_TH_MM": base["REPR_THICKNESS"].sum() * IN_TO_MM,
            "N_AC": int((ac["REPR_THICKNESS"].fillna(0) > 0).sum()),
            "N_BASE": int((base["REPR_THICKNESS"].fillna(0) > 0).sum()),
            "HAS_BOUND_BASE": bool(g["LAYER_TYPE"].isin({"TB", "TS"}).any()),
        })

    out = s.groupby(keys).apply(agg).reset_index()
    return out


# --------------------------------------------------------------------------------------
# Binder properties (INV_PMA_ASPHALT) — viscosity-grade asphalts (GPS-1/SPS-1 are pre-Superpave)
# --------------------------------------------------------------------------------------
# ASPHALT_GRADE codes -> ordinal stiffness rank (viscosity AC grades + a few pen grades).
# Higher rank = stiffer binder.  Codes from LTPP code list (AASHTO M226 viscosity grades).
_GRADE_RANK = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5,            # AC-2.5/5/10/20/40 (viscosity-graded)
               6: 2, 7: 3, 8: 4,                          # AR grades (approximate stiffness)
               9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 4}   # pen / AR-4000 etc. (approx)


def load_fwd_temp():
    """Per-(STATE_CODE, SHRP_ID, TEST_DATE) FWD surface temperature in Celsius.

    SURFACE_TEMP (BAKCAL_BASIN) is in Fahrenheit; FWD_PASS -> TEST_DATE via BAKCAL_PASS.  Used to
    attach a per-survey effective temperature (nearest FWD pass in time) sharper than the seasonal
    climatology.  Returns columns: STATE_CODE, SHRP_ID, TEST_DATE, T_FWD_C."""
    bb = load_bakcal_csv("BAKCAL_BASIN.csv")
    bb["FWD_PASS"] = pd.to_numeric(bb["FWD_PASS"], errors="coerce").astype("Int64")
    bb["SURFACE_TEMP"] = pd.to_numeric(bb["SURFACE_TEMP"], errors="coerce")
    bb = bb[bb["SURFACE_TEMP"].between(-20, 160)]                    # drop sentinels/bad rows (F)
    t = bb.groupby(SEC + ["FWD_PASS"], as_index=False)["SURFACE_TEMP"].median()
    bpass = load_bakcal_csv("BAKCAL_PASS.csv")
    bpass["FWD_PASS"] = pd.to_numeric(bpass["FWD_PASS"], errors="coerce").astype("Int64")
    bpass["TEST_DATE"] = pd.to_datetime(bpass.get("TEST_DATE"), errors="coerce")
    t = t.merge(bpass[SEC + ["FWD_PASS", "TEST_DATE"]].drop_duplicates(SEC + ["FWD_PASS"]),
                on=SEC + ["FWD_PASS"], how="left").dropna(subset=["TEST_DATE"])
    t["T_FWD_C"] = (t["SURFACE_TEMP"] - 32.0) * 5.0 / 9.0
    return t[SEC + ["TEST_DATE", "T_FWD_C"]].sort_values(SEC + ["TEST_DATE"])


def load_binder():
    """Per-(STATE_CODE, SHRP_ID) surface-AC binder properties.

    Returns columns: LOG_VISC_140 (log10 viscosity @140F, poise), PEN_77 (penetration @77F),
    SOFT_PT (ring&ball softening point), BINDER_GRADE_RANK (ordinal stiffness).  Picks the
    top AC layer (min LAYER_NO).  Prefers ORIG_* (as-supplied), falls back to LAB_*.
    """
    b = load_cache("INV_PMA_ASPHALT")
    b["LAYER_NO"] = pd.to_numeric(b.get("LAYER_NO"), errors="coerce")
    for c in ["ASPHALT_GRADE", "ORIG_ASPHALT_VISCOSITY_140", "LAB_VISCOSITY_140",
              "ORIG_PENETRATION_77", "LAB_PENETRATION_77", "ORIG_RING_BALL_SOFTENING_PT"]:
        if c in b.columns:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.sort_values("LAYER_NO").groupby(SEC, as_index=False).first()
    visc = b["ORIG_ASPHALT_VISCOSITY_140"].fillna(b.get("LAB_VISCOSITY_140"))
    pen = b["ORIG_PENETRATION_77"].fillna(b.get("LAB_PENETRATION_77"))
    out = pd.DataFrame({
        "STATE_CODE": b["STATE_CODE"], "SHRP_ID": b["SHRP_ID"],
        "LOG_VISC_140": np.log10(visc.where(visc > 0)),
        "PEN_77": pen,
        "SOFT_PT": b.get("ORIG_RING_BALL_SOFTENING_PT"),
        "BINDER_GRADE_RANK": b["ASPHALT_GRADE"].map(_GRADE_RANK),
    })
    return out


# --------------------------------------------------------------------------------------
# Phase 0: corrected backcalculated moduli (AC / base / subgrade) per FWD pass
# --------------------------------------------------------------------------------------
def load_backcalc_moduli(verify=True):
    """
    Recover per-(STATE_CODE, SHRP_ID, FWD_PASS) layer moduli (MPa) despite the corrupted
    BC_LAYER_NO column, by exploiting top-down row ordering within each pass:
        row 1  -> AC surface (stiffest in flexible sections)
        row 2  -> base
        last   -> subgrade
    Also attaches the FWD test date (BAKCAL_PASS) and best-fit subgrade k-modulus
    (BAKCAL_BEST_FIT_SECTION_MASTER.AVG_K_VALUE) where available.

    Columns returned: STATE_CODE, SHRP_ID, FWD_PASS, TEST_DATE,
        E_AC_BC_MPA, E_BASE_BC_MPA, E_SUBG_BC_MPA, E_SUBG_K_MPA, N_BC_LAYERS, AC_IS_STIFFEST
    """
    bmod = load_bakcal_csv("BAKCAL_MODULUS_SECTION_LAYER.csv")
    bmod["AVG_MODULUS"] = pd.to_numeric(bmod["AVG_MODULUS"], errors="coerce")
    bmod["FWD_PASS"] = pd.to_numeric(bmod["FWD_PASS"], errors="coerce").astype("Int64")
    bmod = bmod.dropna(subset=["AVG_MODULUS"])
    # CSV row order == BC_LAYER_NO ascending (top-down). Preserve it via a stable position index.
    bmod = bmod.reset_index(drop=True)
    bmod["POS"] = bmod.groupby(SEC + ["FWD_PASS"]).cumcount()

    def per_pass(g):
        g = g.sort_values("POS")
        mods = g["AVG_MODULUS"].values
        n = len(mods)
        e_ac = mods[0]
        e_base = mods[1] if n >= 3 else np.nan          # need >=3 layers to have a distinct base
        e_subg = mods[-1] if n >= 2 else np.nan
        return pd.Series({
            "E_AC_BC_MPA": e_ac * KSI_TO_MPA,
            "E_BASE_BC_MPA": e_base * KSI_TO_MPA,
            "E_SUBG_BC_MPA": e_subg * KSI_TO_MPA,
            "N_BC_LAYERS": n,
            "AC_IS_STIFFEST": bool(e_ac >= np.nanmax(mods)),
        })

    mod = bmod.groupby(SEC + ["FWD_PASS"]).apply(per_pass).reset_index()

    # FWD test date
    bpass = load_bakcal_csv("BAKCAL_PASS.csv")
    bpass["FWD_PASS"] = pd.to_numeric(bpass["FWD_PASS"], errors="coerce").astype("Int64")
    if "TEST_DATE" in bpass.columns:
        bpass["TEST_DATE"] = pd.to_datetime(bpass["TEST_DATE"], errors="coerce")
    mod = mod.merge(bpass[SEC + ["FWD_PASS", "TEST_DATE"]].drop_duplicates(SEC + ["FWD_PASS"]),
                    on=SEC + ["FWD_PASS"], how="left")

    # best-fit subgrade k-modulus (psi -> MPa)
    bf = load_bakcal_csv("BAKCAL_BEST_FIT_SECTION_MASTER.csv")
    bf["FWD_PASS"] = pd.to_numeric(bf["FWD_PASS"], errors="coerce").astype("Int64")
    bf["AVG_K_VALUE"] = pd.to_numeric(bf["AVG_K_VALUE"], errors="coerce")
    bf["E_SUBG_K_MPA"] = bf["AVG_K_VALUE"] * KSI_TO_MPA   # AVG_K_VALUE is in ksi (composite modulus units)
    mod = mod.merge(bf[SEC + ["FWD_PASS", "E_SUBG_K_MPA"]].drop_duplicates(SEC + ["FWD_PASS"]),
                    on=SEC + ["FWD_PASS"], how="left")

    if verify:
        frac_stiff = mod["AC_IS_STIFFEST"].mean()
        print(f"  [verify] passes: {len(mod)} | AC-is-stiffest-row: {100*frac_stiff:.1f}% "
              f"| median E_AC={mod.E_AC_BC_MPA.median():.0f} MPa, "
              f"E_base={mod.E_BASE_BC_MPA.median():.0f}, E_subg={mod.E_SUBG_BC_MPA.median():.0f}")
    return mod


# --------------------------------------------------------------------------------------
# Self-test (Phase 0 checkpoint)
# --------------------------------------------------------------------------------------
def _selftest():
    print("Phase 0 self-test: layer structure ...")
    ls = load_layer_structure()
    flex = ls[(ls.AC_TH_MM > 0)]
    print(f"  layer-structure rows: {len(ls)} | with AC: {len(flex)} | "
          f"median AC={flex.AC_TH_MM.median():.0f} mm, base={flex.BASE_TH_MM.median():.0f} mm | "
          f"bound-base sections: {int(ls.HAS_BOUND_BASE.sum())}")

    print("Phase 0 self-test: corrected backcalc moduli ...")
    mod = load_backcalc_moduli(verify=True)
    # physical-plausibility checks
    ok_ac = mod.E_AC_BC_MPA.between(500, 40000).mean()
    ok_base = mod.E_BASE_BC_MPA.between(20, 2000).mean()
    print(f"  E_AC in [500,40000] MPa: {100*ok_ac:.1f}%  |  "
          f"E_base in [20,2000] MPa: {100*ok_base:.1f}%")
    # spot check the cross-validated section
    spot = mod[(mod.STATE_CODE == 42) & (mod.SHRP_ID == "1597") & (mod.FWD_PASS == 20)]
    if len(spot):
        r = spot.iloc[0]
        print(f"  spot (42,1597,pass20): E_AC={r.E_AC_BC_MPA:.0f}  E_base={r.E_BASE_BC_MPA:.0f}  "
              f"E_subg={r.E_SUBG_BC_MPA:.0f} MPa  (expect AC~7960, base~128)")
    print("Phase 0 OK.")


if __name__ == "__main__":
    _selftest()
