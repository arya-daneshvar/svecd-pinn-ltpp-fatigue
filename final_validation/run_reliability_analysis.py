#!/usr/bin/env python3
"""Operational reliability by forecast horizon, observed severity, and calibration."""
from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
from validation_utils import *

MODELS=["Frozen pure PINN","XGBoost","Anchored persistence"]
YEAR_LABELS=["<=2 y",">2-4 y",">4-6 y",">6-10 y",">10 y"]
STEP_LABELS=["1 survey","2 surveys","3 surveys","4+ surveys"]

def add_strata(d):
 d=d.copy(); d["horizon_year_bin"]=pd.cut(d.horizon_years,[-np.inf,2,4,6,10,np.inf],labels=YEAR_LABELS,right=True)
 d["step_bin"]=pd.cut(d.steps_ahead,[0,1,2,3,np.inf],labels=STEP_LABELS,right=True)
 d["observed_severity"]=dominant_state(d,"y")
 # Anchor dominance with higher severity winning ties.
 tmp=d.rename(columns={f"anchor_{s}":f"y_{s}" for s in STATE_NAMES})
 d["anchor_condition"]=dominant_state(tmp,"y").str.replace("-dominant","",regex=False)
 d["total_cracking_bin"]=pd.cut(d.y_total,[-1e-12,0,1,5,10,np.inf],labels=["0%",">0-1%",">1-5%",">5-10%",">10%"],include_lowest=True)
 return d

def subgroup_row(d,kind,label):
 mv=metric_values(d); y=d.y_total.to_numpy(); p=d.p_total.to_numpy(); r=p-y
 out={"stratification":kind,"subgroup":str(label),"protocol":d.protocol.iloc[0],"model":d.model.iloc[0],
      "n_rows":len(d),"n_sections":d.SECTION_ID.nunique(),"median_absolute_error":float(np.median(np.abs(r))),
      "mean_signed_error":float(np.mean(r))}
 out.update({k:v for k,v in mv.items() if k in ["MAE_total","RMSE_total","R2log_total","R2_total","R2log_LOW","R2log_MOD","R2log_HIGH","MAE_LOW","MAE_MOD","MAE_HIGH"]})
 return out

def calibration_bins(d,method):
 z=d.copy()
 if method=="equal_frequency":
  # duplicates='drop' is recorded; actual bin counts are reported.
  z["bin"]=pd.qcut(z.p_onset,q=10,duplicates="drop")
 else: z["bin"]=pd.cut(z.p_onset,np.linspace(0,1,11),include_lowest=True)
 rows=[]
 for b,g in z.groupby("bin",observed=True):
  rows.append({"protocol":g.protocol.iloc[0],"model":g.model.iloc[0],"method":method,"bin":str(b),"n":len(g),"n_sections":g.SECTION_ID.nunique(),
    "mean_predicted_probability":g.p_onset.mean(),"observed_onset_rate":g.y_onset.mean(),"min_probability":g.p_onset.min(),"max_probability":g.p_onset.max()})
 return rows

def run(smoke=False,resume=True,n_boot=1000):
 required=OUT_REL/"horizon_metrics.csv"
 if resume and required.exists() and not smoke: return pd.read_csv(required)
 t0=time.time(); pred=generate_protocol_predictions(smoke=smoke,resume=resume)
 d=add_strata(pred[pred.model.isin(MODELS)&pred.protocol.isin(["temporal","first_anchor","rolling_anchor"])].copy())
 horizon=[]; severity=[]; anchor=[]; counts=[]
 specs=[("horizon_year_bin",YEAR_LABELS,"horizon"),("step_bin",STEP_LABELS,"survey_steps"),
        ("observed_severity",["no cracking","low-dominant","moderate-dominant","high-dominant"],"observed_severity"),
        ("anchor_condition",["no cracking","low","moderate","high"],"anchor_condition"),
        ("total_cracking_bin",["0%",">0-1%",">1-5%",">5-10%",">10%"],"total_cracking")]
 for col,levels,kind in specs:
  for (protocol,model),gm in d.groupby(["protocol","model"]):
   for level in levels:
    g=gm[gm[col].astype(str)==str(level)]
    if not len(g): continue
    r=subgroup_row(g,kind,level)
    counts.append({"stratification":kind,"subgroup":str(level),"protocol":protocol,"model":model,"n_rows":len(g),"n_sections":g.SECTION_ID.nunique()})
    if kind in {"horizon","survey_steps"}: horizon.append(r)
    elif kind=="anchor_condition": anchor.append(r)
    else: severity.append(r)
 if smoke: return pd.DataFrame(horizon)
 hdf=pd.DataFrame(horizon); sdf=pd.DataFrame(severity); adf=pd.DataFrame(anchor)
 atomic_csv(hdf,OUT_REL/"horizon_metrics.csv"); atomic_csv(sdf,OUT_REL/"severity_stratified_metrics.csv"); atomic_csv(adf,OUT_REL/"anchor_state_stratified_metrics.csv")
 atomic_csv(pd.DataFrame(counts),OUT_REL/"reliability_sample_counts.csv")
 # Clustered percentile intervals for key subgroup metrics.
 rng=np.random.default_rng(BOOTSTRAP_SEED+2); ci=[]
 for col,levels,kind in specs[:4]:
  for (protocol,model),gm in d.groupby(["protocol","model"]):
   for level in levels:
    g=gm[gm[col].astype(str)==str(level)]; secs=np.array(g.SECTION_ID.unique())
    if len(g)<3 or len(secs)<2: continue
    vals={m:[] for m in ["MAE_total","RMSE_total","R2log_total","R2log_HIGH"]}
    onset_order=np.argsort(g.p_onset.to_numpy(),kind="mergesort")
    sec_to_pos={s:i for i,s in enumerate(secs)}; row_sec=np.array([sec_to_pos[s] for s in g.SECTION_ID])
    for _ in range(n_boot):
     counts=rng.multinomial(len(secs),np.full(len(secs),1/len(secs))); weights=counts[row_sec].astype(float)
     mv=metric_values(g,weights,onset_order)
     for m in vals: vals[m].append(mv[m])
    obs=metric_values(g)
    for m,v in vals.items():
     lo,hi,ninv=bootstrap_ci(v); ci.append({"stratification":kind,"subgroup":str(level),"protocol":protocol,"model":model,"metric":m,
       "observed":obs[m],"ci_lower":lo,"ci_upper":hi,"n_boot":n_boot,"n_invalid":ninv})
 atomic_csv(pd.DataFrame(ci),OUT_REL/"horizon_metrics_bootstrap_ci.csv")
 # Onset calibration; equal-frequency primary, equal-width sensitivity.
 bins=[]; cm=[]
 for (protocol,model),g in d.groupby(["protocol","model"]):
  for method in ["equal_frequency","equal_width"]: bins.extend(calibration_bins(g,method))
  m=calibration_metrics(g); primary=[x for x in bins if x["protocol"]==protocol and x["model"]==model and x["method"]=="equal_frequency"]
  m["ECE_equal_frequency"]=sum(x["n"]*abs(x["observed_onset_rate"]-x["mean_predicted_probability"]) for x in primary)/sum(x["n"] for x in primary)
  m.update({"protocol":protocol,"model":model,"n_rows":len(g),"n_sections":g.SECTION_ID.nunique(),"onset_threshold":"observed total cracking > 0%"}); cm.append(m)
 atomic_csv(pd.DataFrame(bins),OUT_REL/"onset_calibration_bins.csv"); atomic_csv(pd.DataFrame(cm),OUT_REL/"onset_calibration_metrics.csv")
 # Prespecified catastrophic error threshold: 10 percentage points on 0-100 target scale.
 high=[]
 for (protocol,model),g in d.groupby(["protocol","model"]):
  highdom=g[g.observed_severity=="high-dominant"]; r=g.p_total-g.y_total; rh=highdom.p_HIGH-highdom.y_HIGH
  high.append({"protocol":protocol,"model":model,"n_high_dominant":len(highdom),"n_high_sections":highdom.SECTION_ID.nunique(),
    "high_underprediction_fraction":float(np.mean(rh<0)) if len(rh) else np.nan,"high_mean_signed_error":float(np.mean(rh)) if len(rh) else np.nan,
    "total_abs_error_p90":float(np.percentile(abs(r),90)),"total_abs_error_p95":float(np.percentile(abs(r),95)),
    "catastrophic_error_threshold_pct_points":10.0,"catastrophic_error_rate":float(np.mean(abs(r)>10))})
 atomic_csv(pd.DataFrame(high),OUT_REL/"high_severity_error_summary.csv")
 atomic_json({"source":str(PREDICTIONS.relative_to(ROOT)),"models":MODELS,"onset_threshold":"total cracking > 0%",
  "horizon_year_bins":YEAR_LABELS,"survey_step_bins":STEP_LABELS,"dominant_rule":"largest disjoint state; ties assigned to higher severity",
  "total_bins":["0%",">0-1%",">1-5%",">5-10%",">10%"],"min_r2":"n>=3 and observed variance >1e-12",
  "bootstrap_replicates":n_boot,"bootstrap_seed":BOOTSTRAP_SEED+2,"catastrophic_threshold":"absolute total-cracking error >10 percentage points; prespecified as a large error on the 0-100% area scale",
  "runtime_seconds":time.time()-t0},OUT_REL/"reliability_configuration.json")
 append_log(f"reliability complete runtime={time.time()-t0:.1f}s")
 return hdf

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--smoke-test",action="store_true"); ap.add_argument("--resume",action="store_true"); ap.add_argument("--bootstrap-replicates",type=int,default=1000)
 a=ap.parse_args(); run(a.smoke_test,a.resume,a.bootstrap_replicates)
if __name__=="__main__": main()
