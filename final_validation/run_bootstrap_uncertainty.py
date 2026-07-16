#!/usr/bin/env python3
"""Paired pavement-section cluster bootstrap for all frozen protocols."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import pandas as pd
from validation_utils import *

ERROR_METRICS={"MAE_total","RMSE_total","MAE_LOW","MAE_MOD","MAE_HIGH","Onset_Brier"}

def run(n_boot:int=2000, smoke:bool=False, resume:bool=True):
    t0=time.time(); n_boot=20 if smoke else n_boot
    pred=generate_protocol_predictions(smoke=smoke,resume=resume)
    limited=generate_limited_predictions(smoke=smoke,resume=resume)
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    dist=[]; invalid=[]; manifests=[]; pairdist=[]
    datasets=[("protocol",pred),("limited",limited)]
    for dtype,data in datasets:
        group_cols=["protocol"] if dtype=="protocol" else ["protocol","train_fraction","repeat"]
        for keys,g in data.groupby(group_cols,dropna=False):
            keys=(keys,) if not isinstance(keys,tuple) else keys
            meta=dict(zip(group_cols,keys)); models=sorted(g.model.unique())
            # Exact common row intersection is enforced for every comparison.
            common=set.intersection(*[set(g[g.model==m].row_id) for m in models])
            gg=g[g.row_id.isin(common)].copy(); sections=np.array(sorted(gg.SECTION_ID.unique()))
            model_data={m:gg[gg.model==m].reset_index(drop=True) for m in models}
            onset_orders={m:np.argsort(model_data[m].p_onset.to_numpy(),kind="mergesort") for m in models}
            for m in models:
                dm=model_data[m]; manifests.append({**meta,"sample_type":dtype,"model":m,
                    "n_rows":len(dm),"n_sections":dm.SECTION_ID.nunique(),
                    "row_id_sha256":hashlib.sha256(",".join(map(str,sorted(dm.row_id))).encode()).hexdigest()})
            for b in range(n_boot):
                sampled=rng.choice(sections,size=len(sections),replace=True)
                vals={}
                for m in models:
                    dm=model_data[m]
                    ids,counts=np.unique(sampled,return_counts=True)
                    mult=pd.Series(counts,index=ids)
                    weights=dm.SECTION_ID.map(mult).fillna(0).to_numpy(float)
                    mv=metric_values(dm,weights,onset_orders[m]); vals[m]=mv
                    for metric,value in mv.items():
                        dist.append({**meta,"sample_type":dtype,"model":m,"replicate":b,"metric":metric,"value":value})
                        if not np.isfinite(value): invalid.append({**meta,"sample_type":dtype,"model":m,"replicate":b,"metric":metric,"reason":"insufficient variation or undefined"})
                if "Frozen pure PINN" in vals:
                    for comp in models:
                        if comp=="Frozen pure PINN": continue
                        for metric,pv in vals["Frozen pure PINN"].items():
                            cv=vals[comp].get(metric,np.nan)
                            # Positive always means PINN advantage.
                            diff=(cv-pv) if metric in ERROR_METRICS else (pv-cv)
                            pairdist.append({**meta,"sample_type":dtype,"comparator":comp,"replicate":b,"metric":metric,"advantage":diff})
    dd=pd.DataFrame(dist); pdif=pd.DataFrame(pairdist)
    if smoke: return dd,pdif,pred
    atomic_csv(dd,OUT_BOOT/"bootstrap_metric_distributions.csv")
    summary=[]
    # observed metrics come from original un-resampled samples
    for keys,g in dd.groupby(["sample_type","protocol","train_fraction","repeat","model","metric"],dropna=False):
        meta=dict(zip(["sample_type","protocol","train_fraction","repeat","model","metric"],keys))
        vals=g.value.to_numpy(float); lo,hi,ninv=bootstrap_ci(vals)
        source=limited if meta["sample_type"]=="limited" else pred
        q=source[(source.protocol==meta["protocol"])&(source.model==meta["model"])]
        if meta["sample_type"]=="limited": q=q[(q.train_fraction==meta["train_fraction"])&(q.repeat==meta["repeat"])]
        observed=metric_values(q).get(meta["metric"],np.nan)
        summary.append({**meta,"observed":observed,"bootstrap_mean":float(np.nanmean(vals)),"ci_lower":lo,"ci_upper":hi,
                        "n_boot":len(vals),"n_invalid":ninv,"invalid_fraction":ninv/len(vals)})
    atomic_csv(pd.DataFrame(summary),OUT_BOOT/"bootstrap_metric_summary.csv")
    atomic_csv(pdif,OUT_BOOT/"paired_model_differences.csv")
    ps=[]
    for keys,g in pdif.groupby(["sample_type","protocol","train_fraction","repeat","comparator","metric"],dropna=False):
        meta=dict(zip(["sample_type","protocol","train_fraction","repeat","comparator","metric"],keys)); v=g.advantage.to_numpy(float)
        lo,hi,ninv=bootstrap_ci(v); vv=v[np.isfinite(v)]
        source=limited if meta["sample_type"]=="limited" else pred
        a=source[(source.protocol==meta["protocol"])&(source.model=="Frozen pure PINN")]
        c=source[(source.protocol==meta["protocol"])&(source.model==meta["comparator"])]
        if meta["sample_type"]=="limited":
            for q in (a,c): pass
            a=a[(a.train_fraction==meta["train_fraction"])&(a.repeat==meta["repeat"])]
            c=c[(c.train_fraction==meta["train_fraction"])&(c.repeat==meta["repeat"])]
        assert_identical_rows(a,c); ma=metric_values(a)[meta["metric"]]; mc=metric_values(c)[meta["metric"]]
        obs=(mc-ma) if meta["metric"] in ERROR_METRICS else (ma-mc)
        pbetter=float(np.mean(vv>0)) if len(vv) else np.nan
        p2=float(min(1,2*min(np.mean(vv<=0),np.mean(vv>=0)))) if len(vv) else np.nan
        ps.append({**meta,"observed_difference":obs,"bootstrap_mean_difference":float(np.mean(vv)) if len(vv) else np.nan,
                   "ci_lower":lo,"ci_upper":hi,"proportion_pinn_better":pbetter,
                   "two_sided_empirical_probability":p2,"n_valid":len(vv),"n_invalid":ninv,
                   "orientation":"positive = frozen PINN advantage"})
    atomic_csv(pd.DataFrame(ps),OUT_BOOT/"paired_model_differences_summary.csv")
    invalid_cols=["sample_type","protocol","train_fraction","repeat","model","replicate","metric","reason"]
    atomic_csv(pd.DataFrame(invalid,columns=invalid_cols),OUT_BOOT/"bootstrap_invalid_replicates.csv")
    atomic_csv(pd.DataFrame(manifests),OUT_BOOT/"evaluation_sample_manifest.csv")
    atomic_json({"method":"paired cluster percentile bootstrap","cluster":"SECTION_ID","replicates":n_boot,
                 "seed":BOOTSTRAP_SEED,"ci":[2.5,97.5],"multiplicity":"row expansion per sampled cluster draw",
                 "runtime_seconds":time.time()-t0},OUT_BOOT/"bootstrap_configuration.json")
    append_log(f"bootstrap complete runtime={time.time()-t0:.1f}s")
    return dd,pdif,pred

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bootstrap-replicates",type=int,default=2000)
    ap.add_argument("--smoke-test",action="store_true"); ap.add_argument("--resume",action="store_true")
    a=ap.parse_args(); run(a.bootstrap_replicates,a.smoke_test,a.resume)
if __name__=="__main__": main()
