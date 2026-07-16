#!/usr/bin/env python3
"""Create journal-readable raster (600 dpi) and vector PDF validation figures."""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR","/private/tmp")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from validation_utils import *

COLORS={"Frozen pure PINN":"#1f77b4","XGBoost":"#d62728","CatBoost":"#9467bd","HistGradientBoosting":"#2ca02c","Anchored persistence":"#555555"}
MODELS=["Frozen pure PINN","XGBoost","Anchored persistence"]
YEAR_LABELS=["<=2 y",">2-4 y",">4-6 y",">6-10 y",">10 y"]
def save(fig,path):
 fig.tight_layout(); fig.savefig(path,dpi=600,bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
def line(ax,d,x,y="mean",hue="model",ylabel=None):
 for name,g in d.groupby(hue):
  g=g.sort_values(x); ax.plot(g[x],g[y],marker="o",label=name,color=COLORS.get(name))
  if {"ci_lower","ci_upper"}.issubset(g): ax.fill_between(g[x].astype(float),g.ci_lower,g.ci_upper,alpha=.15,color=COLORS.get(name))
 ax.set_ylabel(ylabel or y); ax.grid(alpha=.2)

def bootstrap_plots():
 s=pd.read_csv(OUT_BOOT/"bootstrap_metric_summary.csv"); p=pd.read_csv(OUT_BOOT/"paired_model_differences_summary.csv")
 q=s[(s.sample_type=="protocol")&s.metric.isin(["R2log_total","MAE_total"])&s.model.isin(COLORS)]
 fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=False)
 for ax,(protocol,g) in zip(axes.ravel(),q.groupby("protocol")):
  gg=g[g.metric=="R2log_total"].sort_values("observed"); y=np.arange(len(gg))
  ax.errorbar(gg.observed,y,xerr=[gg.observed-gg.ci_lower,gg.ci_upper-gg.observed],fmt="o",capsize=3)
  ax.set_yticks(y,gg.model); ax.set_title(protocol.replace('_',' ')); ax.set_xlabel("R²log total (95% CI)"); ax.grid(axis="x",alpha=.2)
 save(fig,OUT_BOOT/"main_metrics_with_95ci.png")
 q=p[(p.sample_type=="protocol")&(p.metric=="R2log_total")&p.comparator.isin(COLORS)]
 fig,ax=plt.subplots(figsize=(9,6)); labs=q.protocol.str.replace('_',' ')+" | "+q.comparator; y=np.arange(len(q))
 ax.errorbar(q.observed_difference,y,xerr=[q.observed_difference-q.ci_lower,q.ci_upper-q.observed_difference],fmt="o",capsize=3)
 ax.axvline(0,color="black",lw=1); ax.set_yticks(y,labs); ax.set_xlabel("PINN advantage in R²log total"); ax.grid(axis="x",alpha=.2)
 save(fig,OUT_BOOT/"paired_differences_forest_plot.png")
 q=s[(s.sample_type=="limited")&(s.metric=="R2log_total")&s.model.isin(["Frozen pure PINN","XGBoost","HistGradientBoosting"])].groupby(["train_fraction","model"],as_index=False).agg(observed=("observed","mean"),ci_lower=("ci_lower","mean"),ci_upper=("ci_upper","mean"))
 fig,ax=plt.subplots(figsize=(7,4.5))
 for name,g in q.groupby("model"):
  g=g.sort_values("train_fraction"); ax.errorbar(100*g.train_fraction,g.observed,yerr=[g.observed-g.ci_lower,g.ci_upper-g.observed],marker="o",capsize=3,label=name,color=COLORS.get(name))
 ax.set_xlabel("Training sections retained (%)"); ax.set_ylabel("R²log total"); ax.legend(fontsize=8); ax.grid(alpha=.2)
 save(fig,OUT_BOOT/"limited_data_retention_with_95ci.png")

def robustness_plots():
 s=pd.read_csv(OUT_ROBUST/"robustness_summary.csv"); r=pd.read_csv(OUT_ROBUST/"robustness_retention_summary.csv")
 specs=[("training_label_noise","label_noise_robustness.png"),
        ("anchor_noise","anchor_noise_robustness.png"),
        ("feature_missingness","feature_group_missingness_robustness.png"),
        ("combined","combined_degradation_robustness.png")]
 for family,file in specs:
  q=s[s.degradation_family.str.startswith(family)&(s.metric=="R2log_total")&s.model.isin(COLORS)]
  fig,ax=plt.subplots(figsize=(8,4.8))
  if family=="feature_missingness":
   for (fam,name),g in q.groupby(["degradation_family","model"]): ax.plot(g.axis_value,g["mean"],marker="o",label=f"{name}: {fam.split(':')[-1]}",alpha=.8)
  elif family=="combined":
   z=q.groupby("model",as_index=False)["mean"].mean(); ax.bar(z.model,z["mean"],color=[COLORS.get(x) for x in z.model]); ax.tick_params(axis='x',rotation=25)
  else: line(ax,q,"axis_value",ylabel="R²log total")
  ax.set_title(family.replace('_',' ')); ax.set_xlabel("Degradation level"); ax.legend(fontsize=7,ncol=2) if family!="combined" else None
  save(fig,OUT_ROBUST/file)
 q=r[(r.degradation_family=="training_label_noise")&(r.metric=="R2log_total")&r.model.isin(COLORS)]
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 line(axes[0],q,"axis_value",ylabel="Absolute R²log total"); line(axes[1],q,"axis_value",y="retention_percent",ylabel="Retention from clean (%)")
 axes[0].set_xlabel("Noise SD (percentage points)"); axes[1].set_xlabel("Noise SD (percentage points)"); axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
 save(fig,OUT_ROBUST/"absolute_vs_relative_robustness.png")

def reliability_plots():
 h=pd.read_csv(OUT_REL/"horizon_metrics.csv"); s=pd.read_csv(OUT_REL/"severity_stratified_metrics.csv"); a=pd.read_csv(OUT_REL/"anchor_state_stratified_metrics.csv")
 b=pd.read_csv(OUT_REL/"onset_calibration_bins.csv"); pred=add_pred_strata(pd.read_csv(PREDICTIONS))
 def catplot(data,subgroup,metric,path,title):
  fig,ax=plt.subplots(figsize=(8,4.8)); order=list(dict.fromkeys(data.subgroup))
  for name,g in data.groupby("model"): ax.plot([order.index(x) for x in g.subgroup],g[metric],marker="o",label=name,color=COLORS.get(name))
  ax.set_xticks(range(len(order)),order,rotation=25,ha="right"); ax.set_ylabel(metric); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.2); save(fig,path)
 catplot(h[(h.stratification=="horizon")&(h.protocol=="first_anchor")],"subgroup","MAE_total",OUT_REL/"error_by_forecast_horizon.png","Total error by elapsed horizon")
 catplot(h[(h.stratification=="survey_steps")&(h.protocol=="first_anchor")],"subgroup","R2log_total",OUT_REL/"performance_by_survey_steps_ahead.png","Performance by surveys ahead")
 catplot(s[(s.stratification=="observed_severity")&(s.protocol=="first_anchor")],"subgroup","MAE_total",OUT_REL/"error_by_observed_severity.png","Error by observed severity")
 catplot(a[(a.protocol=="first_anchor")],"subgroup","MAE_total",OUT_REL/"error_by_anchor_condition.png","Error by anchor condition")
 fig,ax=plt.subplots(figsize=(5.5,5)); q=b[(b.method=="equal_frequency")&(b.protocol=="first_anchor")]
 for name,g in q.groupby("model"): ax.plot(g.mean_predicted_probability,g.observed_onset_rate,marker="o",label=name,color=COLORS.get(name))
 ax.plot([0,1],[0,1],'k--',lw=1); ax.set(xlabel="Mean predicted onset probability",ylabel="Observed onset frequency",title="Onset calibration"); ax.legend(fontsize=8); ax.grid(alpha=.2); save(fig,OUT_REL/"onset_calibration_curve.png")
 q=pred[(pred.protocol=="first_anchor")&pred.model.isin(MODELS)]
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for name,g in q.groupby("model"):
  axes[0].scatter(g.y_total,g.p_total-g.y_total,s=8,alpha=.25,label=name,color=COLORS.get(name)); axes[1].scatter(g.horizon_years,abs(g.p_HIGH-g.y_HIGH),s=8,alpha=.25,label=name,color=COLORS.get(name))
 axes[0].axhline(0,color='k',lw=1); axes[0].set(xlabel="Observed total (%)",ylabel="Residual: predicted - observed (%)"); axes[1].set(xlabel="Years since anchor",ylabel="Absolute high-severity error (%)"); axes[0].legend(fontsize=7)
 save(fig,OUT_REL/"high_severity_residual_diagnostics.png")
 # Compact manuscript panel.
 fig,axes=plt.subplots(2,2,figsize=(9,7))
 for name,g in h[(h.stratification=="horizon")&(h.protocol=="first_anchor")].groupby("model"):
  order=YEAR_LABELS; x=[order.index(v) for v in g.subgroup]; axes[0,0].plot(x,g.MAE_total,marker='o',label=name,color=COLORS.get(name)); axes[0,1].plot(x,g.MAE_HIGH,marker='o',label=name,color=COLORS.get(name))
 axes[0,0].set_title("a  Total-cracking MAE"); axes[0,1].set_title("b  High-severity MAE")
 for ax in axes[0]: ax.set_xticks(range(len(YEAR_LABELS)),YEAR_LABELS,rotation=25,ha='right'); ax.set_xlabel("Years since first anchor"); ax.grid(alpha=.2)
 for name,g in q.groupby("model"): pass
 qb=b[(b.method=="equal_frequency")&(b.protocol=="first_anchor")]
 for name,g in qb.groupby("model"): axes[1,0].plot(g.mean_predicted_probability,g.observed_onset_rate,marker='o',label=name,color=COLORS.get(name))
 axes[1,0].plot([0,1],[0,1],'k--',lw=1); axes[1,0].set(title="c  Onset calibration",xlabel="Predicted probability",ylabel="Observed frequency")
 ss=s[(s.stratification=="observed_severity")&(s.protocol=="first_anchor")]; order=["no cracking","low-dominant","moderate-dominant","high-dominant"]
 for name,g in ss.groupby("model"): axes[1,1].plot([order.index(v) for v in g.subgroup],g.MAE_total,marker='o',label=name,color=COLORS.get(name))
 axes[1,1].set_xticks(range(4),["none","low","moderate","high"],rotation=20); axes[1,1].set(title="d  Error by observed severity",ylabel="MAE total (%)")
 axes[0,0].legend(fontsize=7); save(fig,OUT_REL/"figure_reliability_summary.png")

def add_pred_strata(d):
 d["observed_severity"]=dominant_state(d,"y"); return d
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--smoke-test",action="store_true"); ap.add_argument("--resume",action="store_true"); ap.parse_args()
 bootstrap_plots(); robustness_plots(); reliability_plots()
if __name__=="__main__": main()
