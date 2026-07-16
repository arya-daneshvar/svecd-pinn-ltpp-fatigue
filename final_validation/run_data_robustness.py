#!/usr/bin/env python3
"""Prespecified training-label, anchor-noise, and inference-missingness validation."""
from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from validation_utils import *

LEVELS={"clean":0.0,"mild":1.0,"moderate":3.0,"severe":5.0}  # percentage points
ANCHOR_SIGMA={"clean":0.0,"mild":1.0,"moderate":3.0,"severe":5.0}
MASK_LEVELS=[0,.25,.50,.75,1.0]

def feature_groups():
    f=list(fp.tm.FEATURES)
    groups={
      "climate_environment":[c for c in f if c.startswith("CLM_") or c=="T_EFF_C"],
      "traffic_loading":[c for c in f if c in {"N_CYCLES","LOG_N_CYCLES","CUM_ESAL","ESAL_RATE"}],
      "structure_material":[c for c in f if c in {"AC_THICKNESS","BASE_THICKNESS","AIR_VOIDS_AVG_WP","ASPHALT_CONTENT_MEAN","NO_200_PASSING","NO_4_PASSING","LOG_VISC_140","PEN_77","BINDER_GRADE_RANK","RELAX_SLOPE_M","GPR_AC_DIEL_STD_MEAN","GPR_AC_DIEL_MEAN","GPR_AC_THICK_STD_MEAN","GPR_AC_THICK_MEAN_SUM","DRAIN_DRAINAGE_PRESENT","E_AC_BC_MPA","E_BASE_BC_MPA","E_SUBG_BC_MPA"}],
      "mechanistic_svecd_lea":[c for c in f if c in {"EPS_T_MICRO","EPS_R","ESTAR_TEFF_MPA","ESTAR_REF_MPA"}],
      "previous_condition_anchor":[c for c in f if c in {"LAST_OBS_CPCT_L","LAST_OBS_CPCT_M","LAST_OBS_CPCT_H","AGE_AT_LAST_OBS","DT_SINCE_LAST"}],
    }
    return groups

def corrected_noise(y,rng,sigma):
    z=np.clip(y+rng.normal(0,sigma,y.shape),0,100)
    sums=z.sum(1); over=sums>100; z[over]*=(100/sums[over])[:,None]
    return z.astype(np.float32)

def rebuild_train_anchor(X,yall,tr):
    X=X.copy(); pos={c:i for i,c in enumerate(fp.tm.FEATURES)}
    # Frozen representation has cumulative-low last observation only by default.
    if "LAST_OBS_CPCT_L" in pos:
        prev=fp.PREV[tr]; ok=prev>=0; vals=X[tr,pos["LAST_OBS_CPCT_L"]].copy()
        vals[ok]=yall[prev[ok]].sum(1); X[tr,pos["LAST_OBS_CPCT_L"]]=vals
    return X

def row(case,level,seed,model,pred,prob,te,axis_value,protocol="temporal"):
    d=prediction_frame(protocol,model,te,fp.PREV[te],pred,prob)
    mv=metric_values(d); out={"degradation_family":case,"level":level,"seed":seed,"model":model,
        "protocol":protocol,"axis_value":axis_value,"n_rows":len(d),"n_sections":d.SECTION_ID.nunique()}
    out.update(mv); return out

def tabular_predictions(Xtr,Xte,ytr,seed=0): return fit_tabular(Xtr,Xte,ytr,seed,include_catboost=False)

def run(smoke=False,resume=True,seeds=None):
    outpath=OUT_ROBUST/"robustness_all_runs.csv"
    if resume and outpath.exists() and not smoke: return pd.read_csv(outpath)
    t0=time.time(); seeds=[0] if smoke else (seeds or list(range(5))); tr,te=fp.temporal_split()
    partial=outpath.with_name("robustness_all_runs_partial.csv")
    results=(pd.read_csv(partial).to_dict("records") if resume and partial.exists() and not smoke else [])
    done={(r["degradation_family"],str(r["level"]),int(r["seed"]),r["model"],r["protocol"]) for r in results}
    Xclean=fp.tm.build_X(anchor_on=True); manifests=[]
    # A. Training-label degradation. Model initialization is held at seed 0.
    lab_levels={"clean":0.0,"moderate":3.0} if smoke else LEVELS
    for seed in seeds:
      for level,sigma in lab_levels.items():
        rng=np.random.default_rng(BASE_SEED+1000*seed+int(sigma*10)); yall=fp.Y_STATE.copy()
        yall[tr]=corrected_noise(fp.Y_STATE[tr],rng,sigma)
        Xcase=rebuild_train_anchor(Xclean,yall,tr)
        mask_hash=hashlib.sha256(yall[tr].tobytes()).hexdigest(); manifests.append({"family":"training_label_noise","level":level,"seed":seed,"realization_sha256":mask_hash,"shared_across_models":True})
        if ("training_label_noise",str(level),seed,"Frozen pure PINN","temporal") in done: continue
        model,imp,scl,Xall=fit_frozen(tr,te,True,0,yall[tr],Xcase)
        ps,po=predict_frozen_temporal(model,imp,scl,Xclean,te) # test covariates/anchors clean
        results.append(row("training_label_noise",level,seed,"Frozen pure PINN",ps,po,te,sigma))
        for name,(p,prob) in tabular_predictions(Xcase[tr],Xclean[te],yall[tr],0).items(): results.append(row("training_label_noise",level,seed,name,p,prob,te,sigma))
        pp,ppo=persistence_prediction(te,fp.PREV[te],tr); results.append(row("training_label_noise",level,seed,"Anchored persistence",pp,ppo,te,sigma))
        if not smoke: atomic_csv(pd.DataFrame(results),partial)

    # B. Inference-time feature masking: train clean once, then reuse preprocessing/model.
    model,imp,scl,Xall=fit_frozen(tr,te,True,0); groups=feature_groups()
    clean_tabular=fit_tabular_models(Xclean[tr],fp.Y_STATE[tr],0,False)
    group_iter={"traffic_loading":groups["traffic_loading"]} if smoke else groups
    for gname,cols in group_iter.items():
      levs=[0,.5] if smoke else MASK_LEVELS
      for seed in seeds:
       for frac in levs:
        Xcase=Xclean.copy(); rng=np.random.default_rng(BASE_SEED+20000+seed*100+int(frac*100)+sum(map(ord,gname)))
        candidate=np.unique(fp.tm.groups[te]); n=int(round(frac*len(candidate)))
        masked_secs=np.sort(rng.choice(candidate,size=n,replace=False)) if n else np.array([])
        target_rows=te[np.isin(fp.tm.groups[te],masked_secs)]
        # blockwise section masking; anchor rows are included for consistency.
        related=np.unique(np.r_[target_rows,fp.PREV[target_rows][fp.PREV[target_rows]>=0]])
        for c in cols:
            if c in fp.tm.FEATURES: Xcase[related,fp.tm.FEATURES.index(c)]=np.nan
        h=hashlib.sha256((gname+str(frac)+",".join(map(str,masked_secs))).encode()).hexdigest()
        manifests.append({"family":"feature_group_missingness","group":gname,"level":frac,"seed":seed,"realization_sha256":h,"shared_across_models":True,"masking":"section block"})
        ps,po=predict_frozen_temporal(model,imp,scl,Xcase,te)
        if gname=="previous_condition_anchor" and len(target_rows):
            Xe=scl.transform(imp.transform(Xcase[target_rows])).astype(np.float32)
            pabs,_=fp.predict_absolute(model,Xe); loc=np.flatnonzero(np.isin(te,target_rows)); ps[loc]=pabs
        label=("cold_start_conversion" if frac==1 else f"mixed_anchor_cold_start_{frac:g}") if gname=="previous_condition_anchor" and frac>0 else f"{frac:g}"
        results.append(row(f"feature_missingness:{gname}",label,seed,"Frozen pure PINN",ps,po,te,frac))
        for name,(p,prob) in predict_tabular_models(clean_tabular,Xcase[te]).items(): results.append(row(f"feature_missingness:{gname}",label,seed,name,p,prob,te,frac))
        pp,ppo=persistence_prediction(te,fp.PREV[te],tr)
        if gname=="previous_condition_anchor" and len(target_rows):
            loc=np.flatnonzero(np.isin(te,target_rows)); pp[loc]=np.nanmedian(fp.Y_STATE[tr],axis=0); ppo[loc]=np.clip(pp[loc].sum(1)/100,1e-6,1-1e-6)
        results.append(row(f"feature_missingness:{gname}",label,seed,"Anchored persistence",pp,ppo,te,frac))

    # C. Anchor noise under held-out-section first and rolling protocols.
    anchor_levels={"clean":0.0,"moderate":3.0} if smoke else ANCHOR_SIGMA
    gkf=GroupKFold(n_splits=2 if smoke else 5); first=fp.first_indices(); Xfirst=fp.fixed_anchor_matrix(first); Xroll=Xclean
    anchor_acc={}
    def acc(protocol,level,seed,name,p,prob,ev): anchor_acc.setdefault((protocol,level,seed,name),[]).append((ev,p,prob))
    for fold,(tri,held) in enumerate(gkf.split(fp.tm.build_X(False),fp.Y_TOTAL,fp.tm.groups),1):
      ev=np.asarray([i for i in held if first[i]!=i]); mdl,im,sc,_=fit_frozen(tri,ev,True,fold-1)
      for protocol,Xbase,anchors in [("first_anchor",Xfirst,first[ev]),("rolling_anchor",Xroll,fp.PREV[ev])]:
       fitted_tab=fit_tabular_models(Xbase[tri],fp.Y_STATE[tri],fold-1,False)
       for seed in seeds:
        for level,sigma in anchor_levels.items():
         rng=np.random.default_rng(BASE_SEED+30000+fold*1000+seed*10+int(sigma)); noisy=fp.Y_STATE[anchors].copy()
         noisy=corrected_noise(noisy,rng,sigma); Xc=Xbase.copy(); pos={c:i for i,c in enumerate(fp.tm.FEATURES)}
         if "LAST_OBS_CPCT_L" in pos: Xc[ev,pos["LAST_OBS_CPCT_L"]]=noisy.sum(1)
         # Transition must add the same corrupted cumulative anchor, so temporarily swap only anchor states.
         oldS,oldC=fp.Y_STATE,fp.Y_CUM
         try:
          ys=oldS.copy(); ys[anchors]=noisy; fp.Y_STATE=ys; fp.Y_CUM=state_to_cum(ys).astype(np.float32)
          ps,po=predict_frozen_anchor(mdl,im,sc,ev,anchors,Xc)
         finally: fp.Y_STATE,fp.Y_CUM=oldS,oldC
         h=hashlib.sha256(noisy.tobytes()).hexdigest(); manifests.append({"family":"anchor_noise","protocol":protocol,"fold":fold,"level":level,"seed":seed,"realization_sha256":h,"shared_across_models":True})
         acc(protocol,level,seed,"Frozen pure PINN",ps,po,ev)
         for name,(p,prob) in predict_tabular_models(fitted_tab,Xc[ev]).items(): acc(protocol,level,seed,name,p,prob,ev)
         acc(protocol,level,seed,"Anchored persistence",noisy,np.clip(noisy.sum(1)/100,1e-6,1-1e-6),ev)
    for (protocol,level,seed,name),pieces in anchor_acc.items():
      pieces=sorted(pieces,key=lambda x:int(np.min(x[0]))); ev=np.concatenate([x[0] for x in pieces]); order=np.argsort(ev); ev=ev[order]
      ps=np.concatenate([x[1] for x in pieces])[order]; po=np.concatenate([x[2] for x in pieces])[order]
      results.append(row("anchor_noise",level,seed,name,ps,po,ev,ANCHOR_SIGMA[level],protocol))

    # D. Prespecified combined inference degradation on temporal test.
    for seed in seeds:
      Xc=Xclean.copy(); rng=np.random.default_rng(BASE_SEED+40000+seed); anchors=fp.PREV[te]
      noisy=corrected_noise(fp.Y_STATE[anchors],rng,3.0); pos={c:i for i,c in enumerate(fp.tm.FEATURES)}
      if "LAST_OBS_CPCT_L" in pos: Xc[te,pos["LAST_OBS_CPCT_L"]]=noisy.sum(1)
      for gname in ["traffic_loading","climate_environment"]:
       secs=np.unique(fp.tm.groups[te]); selected=rng.choice(secs,size=int(round(.5*len(secs))),replace=False); rr=te[np.isin(fp.tm.groups[te],selected)]
       for c in groups[gname]:
        if c in pos: Xc[rr,pos[c]]=np.nan
      oldS,oldC=fp.Y_STATE,fp.Y_CUM
      try:
       ys=oldS.copy(); ys[anchors]=noisy; fp.Y_STATE=ys; fp.Y_CUM=state_to_cum(ys).astype(np.float32)
       ps,po=predict_frozen_temporal(model,imp,scl,Xc,te)
      finally: fp.Y_STATE,fp.Y_CUM=oldS,oldC
      manifests.append({"family":"combined","level":"moderate_anchor_50pct_traffic_50pct_climate","seed":seed,"realization_sha256":hashlib.sha256(Xc[te].tobytes()).hexdigest(),"shared_across_models":True})
      results.append(row("combined","combined",seed,"Frozen pure PINN",ps,po,te,1.0))
      for name,(p,prob) in predict_tabular_models(clean_tabular,Xc[te]).items(): results.append(row("combined","combined",seed,name,p,prob,te,1.0))
      results.append(row("combined","combined",seed,"Anchored persistence",noisy,np.clip(noisy.sum(1)/100,1e-6,1-1e-6),te,1.0))

    allr=pd.DataFrame(results)
    if smoke: return allr
    atomic_csv(allr,outpath); atomic_csv(pd.DataFrame(manifests),OUT_ROBUST/"corruption_manifest.csv")
    frows=[]
    for g,cols in groups.items():
      for c in cols: frows.append({"feature_group":g,"feature":c,"source":"frozen obs30 list and frozen data dictionary","represented":True})
    atomic_csv(pd.DataFrame(frows),OUT_ROBUST/"feature_group_dictionary.csv")
    metrics=[c for c in allr if c.startswith("R2") or c.startswith("MAE") or c.startswith("Onset_") or c=="RMSE_total"]
    long=allr.melt(id_vars=["degradation_family","level","seed","model","protocol","axis_value","n_rows","n_sections"],value_vars=metrics,var_name="metric",value_name="value")
    summ=long.groupby(["degradation_family","level","model","protocol","axis_value","metric"],dropna=False).value.agg(["mean","std","count"]).reset_index()
    summ["ci_lower"]=summ["mean"]-1.96*summ["std"]/np.sqrt(summ["count"]); summ["ci_upper"]=summ["mean"]+1.96*summ["std"]/np.sqrt(summ["count"])
    atomic_csv(summ,OUT_ROBUST/"robustness_summary.csv")
    clean=(summ[(summ.axis_value==0)].rename(columns={"mean":"clean_mean"})[["degradation_family","model","protocol","metric","clean_mean"]])
    ret=summ.merge(clean,on=["degradation_family","model","protocol","metric"],how="left")
    ret["absolute_change"]=ret["mean"]-ret["clean_mean"]; ret["retention_percent"]=100*ret["mean"]/ret["clean_mean"]
    atomic_csv(ret,OUT_ROBUST/"robustness_retention_summary.csv")
    auc=[]
    for keys,g in summ.groupby(["degradation_family","model","protocol","metric"]):
      g=g.sort_values("axis_value"); x=g.axis_value.to_numpy(float); y=g["mean"].to_numpy(float)
      if len(np.unique(x))>1: auc.append(dict(zip(["degradation_family","model","protocol","metric"],keys),normalized_auc=float(np.trapz(y,x)/(x.max()-x.min()))))
    atomic_csv(pd.DataFrame(auc),OUT_ROBUST/"robustness_auc_summary.csv")
    atomic_json({"seeds":seeds,"label_noise_pct_points":LEVELS,"anchor_noise_pct_points":ANCHOR_SIGMA,"mask_fractions":MASK_LEVELS,
      "label_correction":"clip each disjoint state to [0,100], then proportionally rescale rows whose sum exceeds 100",
      "masking":"section-block masking; clean-training median imputation","combined":"3 pp anchor noise + 50% traffic sections + 50% climate sections","runtime_seconds":time.time()-t0},OUT_ROBUST/"robustness_configuration.json")
    append_log(f"robustness complete runtime={time.time()-t0:.1f}s")
    return allr

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--smoke-test",action="store_true"); ap.add_argument("--resume",action="store_true"); ap.add_argument("--seeds",default="0,1,2,3,4")
 a=ap.parse_args(); run(a.smoke_test,a.resume,[int(x) for x in a.seeds.split(',')])
if __name__=="__main__": main()
