#!/usr/bin/env python3
"""Orchestrate all final-validation packages and stop on major QC failure."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from validation_utils import *

def write_test_report(checks):
 lines=["# Final validation test report","",f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}","",
        "| Test | Status | Major | Detail |","|---|---|---:|---|"]
 for c in checks: lines.append(f"| {c['test']} | {c['status']} | {c['major']} | {c['detail']} |")
 failures=[c for c in checks if c['major'] and c['status']!='PASS']
 lines += ["",f"Major failures: **{len(failures)}**.","",
  "The full experiment is permitted only when this count is zero. Metric reproduction uses the saved 277-row temporal prediction sample and a tolerance of 2×10⁻⁵."]
 (HERE/"FINAL_VALIDATION_TEST_REPORT.md").write_text("\n".join(lines))
 return failures

def call(script,*args):
 cmd=[sys.executable,str(HERE/script),*args]; append_log("CALL "+" ".join(cmd)); subprocess.run(cmd,cwd=HERE,check=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--smoke-test",action="store_true"); ap.add_argument("--resume",action="store_true")
 a=ap.parse_args(); write_base_manifests()
 if a.smoke_test:
  call("run_bootstrap_uncertainty.py","--smoke-test","--resume")
  call("run_data_robustness.py","--smoke-test","--resume")
  call("run_reliability_analysis.py","--smoke-test","--resume")
  pred=pd.read_csv(PREDICTIONS) if PREDICTIONS.exists() else None
  failures=write_test_report(qc_checks(pred));
  if failures: raise SystemExit("major smoke-test failure; full run stopped")
  return
 pred=generate_protocol_predictions(resume=True)
 failures=write_test_report(qc_checks(pred))
 if failures: raise SystemExit("major QC failure; full run stopped")
 call("run_bootstrap_uncertainty.py","--resume","--bootstrap-replicates","2000")
 call("run_data_robustness.py","--resume","--seeds","0,1,2,3,4")
 call("run_reliability_analysis.py","--resume","--bootstrap-replicates","1000")
 call("plot_final_validation.py")
 call("generate_final_reports.py")
 write_test_report(qc_checks(pd.read_csv(PREDICTIONS)))
if __name__=="__main__": main()
