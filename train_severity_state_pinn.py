"""
Pure S-VECD PINN trained on LTPP disjoint severity-state targets.

The standard cumulative target is useful for monotone ordering, but LTPP defines fatigue severity
states as disjoint visual classes:
  low-only, moderate-only, high-only alligator/fatigue cracking area.

This model keeps the ordered cumulative transfer internally:
  E_L = low-or-higher area
  E_M = moderate-or-higher area
  E_H = high area

Then it converts to disjoint states for the data loss:
  low-only      = E_L - E_M
  moderate-only = E_M - E_H
  high-only     = E_H

Transition prediction still anchors cumulative severity-or-higher states:
  E_hat_k = E_obs_prev + relu(E_pinn_k - E_pinn_prev)

Then disjoint states are recovered from E_hat_k. This lets low-only area decline as cracking moves
into moderate/high states, while preserving monotone fatigue progression in cumulative state space.

Run:
  SVECD_TAG=_sdr_dense_state python3 train_severity_state_pinn.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("SVECD_TAG", "_sdr_dense_state")
os.environ.setdefault("SVECD_ANCHOR", "1")
os.environ.setdefault("SVECD_HYBRID", "0")
os.environ.setdefault("SVECD_ENSEMBLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import train_models as tm


PROJ = Path(__file__).resolve().parent
TAG = os.environ.get("SVECD_TAG", "_sdr_dense_state")
OUT_SUFFIX = os.environ.get("STATE_PINN_OUT_SUFFIX", "")
OUT = PROJ / f"outputs_severity_state_pinn{TAG}{OUT_SUFFIX}"
OUT.mkdir(exist_ok=True)
RUN_PARTS = os.environ.get("STATE_PINN_RUN_PARTS", "all").lower()
SELF_ANCHOR_BASELINE = os.environ.get("STATE_PINN_SELF_ANCHOR_BASELINE", "0") == "1"
SEED_OFFSET = int(os.environ.get("STATE_PINN_SEED_OFFSET", "0"))
SPLIT_LOCAL_DENSE = os.environ.get("STATE_PINN_SPLIT_LOCAL_DENSE", "1") == "1"

EPOCHS = int(os.environ.get("STATE_PINN_EPOCHS", "650"))
LR = float(os.environ.get("STATE_PINN_LR", "0.002"))
W_PHYS_LIST = [float(x) for x in os.environ.get("STATE_PINN_W_PHYS_LIST", "0.03,0.1,0.3").split(",")]
W_TRANS_LIST = [float(x) for x in os.environ.get("STATE_PINN_W_TRANS_LIST", "0.5,1,2").split(",")]
W_TRAJ_LIST = [float(x) for x in os.environ.get("STATE_PINN_W_TRAJ_LIST", "0").split(",")]
TRAJ_PAIR_MODE = os.environ.get("STATE_PINN_TRAJ_PAIR_MODE", "first").lower()
TRAJ_MAX_PAIRS = int(os.environ.get("STATE_PINN_TRAJ_MAX_PAIRS", "20000"))
W_MONO = float(os.environ.get("STATE_PINN_W_MONO", "0.1"))
W_ALPHA = float(os.environ.get("STATE_PINN_W_ALPHA", str(tm.W_ALPHA)))
W_ONSET = float(os.environ.get("STATE_PINN_W_ONSET", "0.15"))
W_CRACKED = float(os.environ.get("STATE_PINN_W_CRACKED", "0.5"))
W_TOTAL = float(os.environ.get("STATE_PINN_W_TOTAL", "0.4"))
HUBER_DELTA = float(os.environ.get("STATE_PINN_HUBER_DELTA", "1.0"))
STATE_BAND_W = torch.tensor([float(x) for x in os.environ.get("STATE_PINN_BAND_W", "1.0,1.2,1.5").split(",")], dtype=torch.float32)

STATE_COLS = ["C_PCT_LOW_ONLY", "C_PCT_MOD_ONLY", "C_PCT_HIGH_ONLY"]
CUM_COLS = ["C_PCT_L", "C_PCT_M", "C_PCT_H"]


@dataclass(frozen=True)
class Variant:
    name: str
    w_phys: float
    w_trans: float
    seed: int = 0
    w_traj: float = 0.0


def require_targets() -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in STATE_COLS if c not in tm.df.columns]
    if missing:
        raise RuntimeError(f"missing severity-state columns: {missing}; run build_severity_state_dataset.py")
    y_state = tm.df[STATE_COLS].to_numpy(np.float32)
    y_cum = tm.df[CUM_COLS].to_numpy(np.float32)
    return y_state, y_cum


Y_STATE, Y_CUM = require_targets()
Y_TOTAL = Y_STATE.sum(axis=1)


def previous_index() -> np.ndarray:
    prev = np.full(len(tm.df), -1, dtype=int)
    for _, g in tm.df.reset_index().sort_values(["SECTION_ID", "AGE"]).groupby("SECTION_ID"):
        idx = g["index"].to_numpy()
        prev[idx[1:]] = idx[:-1]
    return prev


PREV = previous_index()


def enforce_order(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0.0, 100.0)
    return np.minimum.accumulate(P, axis=1) if P.ndim == 2 and P.shape[1] > 1 else P


def cum_to_state_np(cum: np.ndarray) -> np.ndarray:
    cum = enforce_order(cum)
    return np.column_stack([
        np.clip(cum[:, 0] - cum[:, 1], 0, 100),
        np.clip(cum[:, 1] - cum[:, 2], 0, 100),
        np.clip(cum[:, 2], 0, 100),
    ])


def state_to_cum_np(state: np.ndarray) -> np.ndarray:
    state = np.clip(state, 0, 100)
    return np.column_stack([
        state[:, 0] + state[:, 1] + state[:, 2],
        state[:, 1] + state[:, 2],
        state[:, 2],
    ]).clip(0, 100)


def cum_to_state_torch(cum: torch.Tensor) -> torch.Tensor:
    cum = torch.clamp(cum, 0.0, 100.0)
    cols = [cum[:, 0]]
    for j in range(1, cum.shape[1]):
        cols.append(torch.minimum(cols[-1], cum[:, j]))
    cum = torch.stack(cols, dim=1)
    return torch.stack([
        torch.clamp(cum[:, 0] - cum[:, 1], min=0.0),
        torch.clamp(cum[:, 1] - cum[:, 2], min=0.0),
        torch.clamp(cum[:, 2], min=0.0),
    ], dim=1)


def temporal_split() -> tuple[np.ndarray, np.ndarray]:
    last_idx = (
        tm.df[tm.df.IS_PROGRESSION == 1]
        .sort_values("AGE")
        .groupby("SECTION_ID")
        .tail(1)
        .index.to_numpy()
    )
    te_mask = np.zeros(len(tm.df), dtype=bool)
    te_mask[last_idx] = True
    return np.where(~te_mask)[0], np.where(te_mask)[0]


def split_local_dense_matrix(tr: np.ndarray, anchor_on: bool) -> np.ndarray:
    """Rebuild selected dense predictors from STRICT_ values using training rows only."""
    Xall = tm.build_X(anchor_on=anchor_on)
    if not SPLIT_LOCAL_DENSE:
        return Xall

    frame = tm.df
    train = frame.iloc[tr].copy()
    apply = frame.copy()
    section_col = "PHYSICAL_SECTION_ID" if "PHYSICAL_SECTION_ID" in frame.columns else "SECTION_ID"
    train_year = pd.to_datetime(train["SURVEY_DATE"], errors="coerce").dt.year
    apply_year = pd.to_datetime(apply["SURVEY_DATE"], errors="coerce").dt.year

    for col_idx, feature in enumerate(tm.FEATURES):
        source = f"STRICT_{feature}"
        if source not in frame.columns:
            continue
        train_values = pd.to_numeric(train[source], errors="coerce")
        result = pd.to_numeric(apply[source], errors="coerce").copy()
        section_median = train_values.groupby(train[section_col]).median()
        result = result.fillna(apply[section_col].map(section_median))
        state_year_median = train_values.groupby(
            [train["STATE_CODE"], train_year], dropna=False
        ).median()
        state_year_fill = pd.Series(
            [state_year_median.get((state, year), np.nan)
             for state, year in zip(apply["STATE_CODE"], apply_year)],
            index=apply.index,
        )
        result = result.fillna(state_year_fill)
        state_median = train_values.groupby(train["STATE_CODE"]).median()
        result = result.fillna(apply["STATE_CODE"].map(state_median))
        global_median = train_values.median()
        result = result.fillna(0.0 if pd.isna(global_median) else global_median)
        Xall[:, col_idx] = result.to_numpy(np.float32)
    return Xall.astype(np.float32)


def scaled_matrices(tr: np.ndarray, te: np.ndarray, anchor_on: bool):
    Xall = split_local_dense_matrix(tr, anchor_on=anchor_on)
    imp = SimpleImputer(strategy="median").fit(Xall[tr])
    scl = StandardScaler().fit(imp.transform(Xall[tr]))
    return (
        scl.transform(imp.transform(Xall[tr])).astype(np.float32),
        scl.transform(imp.transform(Xall[te])).astype(np.float32),
        Xall,
        imp,
        scl,
    )


def transform_indices(Xall: np.ndarray, imp: SimpleImputer, scl: StandardScaler, idx: np.ndarray):
    return scl.transform(imp.transform(Xall[idx])).astype(np.float32)


def anchored_rows_matrix(
    row_idx: np.ndarray, anchor_idx: np.ndarray, base_matrix: np.ndarray | None = None
) -> np.ndarray:
    source = tm.build_X(anchor_on=True) if base_matrix is None else base_matrix
    x = source[row_idx].copy()
    col = {c: tm.FEATURES.index(c) for c in tm.FEATURES}
    for j, suffix in enumerate(["L", "M", "H"]):
        name = f"LAST_OBS_CPCT_{suffix}"
        if name in col:
            x[:, col[name]] = Y_CUM[anchor_idx, j].astype(np.float32)
    if "AGE_AT_LAST_OBS" in col:
        x[:, col["AGE_AT_LAST_OBS"]] = tm.AGEv[anchor_idx].astype(np.float32)
    if "DT_SINCE_LAST" in col:
        x[:, col["DT_SINCE_LAST"]] = (tm.AGEv[row_idx] - tm.AGEv[anchor_idx]).astype(np.float32)
    return x.astype(np.float32)


def trajectory_pairs(global_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    allowed = set(int(i) for i in global_idx)
    anchors, futures = [], []
    for _, g in tm.df.reset_index().sort_values(["SECTION_ID", "AGE"]).groupby("SECTION_ID"):
        idx = [int(i) for i in g["index"].to_numpy() if int(i) in allowed]
        if len(idx) < 2:
            continue
        if TRAJ_PAIR_MODE == "first":
            anchors.extend([idx[0]] * (len(idx) - 1))
            futures.extend(idx[1:])
        elif TRAJ_PAIR_MODE == "long":
            for a_pos, a in enumerate(idx[:-2]):
                for f in idx[a_pos + 2:]:
                    anchors.append(a)
                    futures.append(f)
            if len(idx) > 1:
                anchors.extend([idx[0]] * (len(idx) - 1))
                futures.extend(idx[1:])
        elif TRAJ_PAIR_MODE == "all":
            for a_pos, a in enumerate(idx[:-1]):
                for f in idx[a_pos + 1:]:
                    anchors.append(a)
                    futures.append(f)
        else:
            raise ValueError(f"unknown trajectory pair mode: {TRAJ_PAIR_MODE}")

    anchors = np.asarray(anchors, dtype=int)
    futures = np.asarray(futures, dtype=int)
    if len(futures) > TRAJ_MAX_PAIRS:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(futures), size=TRAJ_MAX_PAIRS, replace=False))
        anchors = anchors[keep]
        futures = futures[keep]
    return anchors, futures


def make_trajectory_data(
    global_idx: np.ndarray,
    imp: SimpleImputer,
    scl: StandardScaler,
    base_matrix: np.ndarray | None = None,
):
    anchors, futures = trajectory_pairs(global_idx)
    if len(futures) == 0:
        return None
    Xfuture = scl.transform(
        imp.transform(anchored_rows_matrix(futures, anchors, base_matrix))
    ).astype(np.float32)
    Xanchor = scl.transform(
        imp.transform(anchored_rows_matrix(anchors, anchors, base_matrix))
    ).astype(np.float32)
    row_w = 1.0 + W_CRACKED * (Y_STATE[futures].sum(axis=1) > 0).astype(np.float32)
    return {
        "Xfuture": Xfuture,
        "Xanchor": Xanchor,
        "y_state": Y_STATE[futures],
        "y_cum": Y_CUM[futures],
        "anchor_cum": Y_CUM[anchors],
        "row_w": row_w.astype(np.float32),
        "n_pairs": int(len(futures)),
    }


def train_variant(
    Xtr: np.ndarray,
    y_state_tr: np.ndarray,
    y_cum_tr: np.ndarray,
    eps_r: np.ndarray,
    dN: np.ndarray,
    sections: np.ndarray,
    ages: np.ndarray,
    tr_global: np.ndarray,
    variant: Variant,
    Xtr_self_anchor: np.ndarray | None = None,
    traj_data: dict | None = None,
):
    torch.manual_seed(variant.seed)
    np.random.seed(variant.seed)

    pinn = tm.SvecdPINN(Xtr.shape[1], n_bands=tm.N_BANDS)
    opt = torch.optim.AdamW(pinn.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    huber = nn.HuberLoss(delta=HUBER_DELTA, reduction="none")
    bce = nn.BCEWithLogitsLoss()

    xb = torch.tensor(Xtr, requires_grad=True)
    xb_self = torch.tensor(Xtr_self_anchor, dtype=torch.float32) if Xtr_self_anchor is not None else None
    ys = torch.tensor(y_state_tr, dtype=torch.float32)
    yc = torch.tensor(y_cum_tr, dtype=torch.float32)
    onset_t = torch.tensor((y_state_tr.sum(axis=1) > 0).astype(np.float32))
    er = torch.tensor(eps_r.astype(np.float32))
    dn = torch.tensor(dN.astype(np.float32))
    alpha_prior = float(np.nanmedian(tm.ALPHA_PRIOR_v[tr_global])) if tm.ALPHA_PRIOR_v is not None else None

    global_to_pos = {int(g): i for i, g in enumerate(tr_global)}
    cur_pos, prev_pos = [], []
    for pos, gidx in enumerate(tr_global):
        p = int(PREV[gidx])
        if p in global_to_pos:
            cur_pos.append(pos)
            prev_pos.append(global_to_pos[p])
    cur_pos = torch.tensor(cur_pos, dtype=torch.long)
    prev_pos = torch.tensor(prev_pos, dtype=torch.long)

    order, inv, prev_idx = tm.segment_index(sections, ages)
    order = torch.tensor(order)
    inv = torch.tensor(inv)
    prev_idx = torch.tensor(prev_idx)
    row_w = 1.0 + W_CRACKED * torch.tensor((y_state_tr.sum(axis=1) > 0).astype(np.float32))
    bw = STATE_BAND_W.view(1, -1)
    if traj_data is not None:
        traj_xf = torch.tensor(traj_data["Xfuture"], dtype=torch.float32)
        traj_xa = torch.tensor(traj_data["Xanchor"], dtype=torch.float32)
        traj_ys = torch.tensor(traj_data["y_state"], dtype=torch.float32)
        traj_yc = torch.tensor(traj_data["y_cum"], dtype=torch.float32)
        traj_anchor_yc = torch.tensor(traj_data["anchor_cum"], dtype=torch.float32)
        traj_row_w = torch.tensor(traj_data["row_w"], dtype=torch.float32)
    else:
        traj_xf = traj_xa = traj_ys = traj_yc = traj_anchor_yc = traj_row_w = None

    def state_huber(pred_state, targ_state, weights=None):
        loss = huber(torch.log1p(pred_state), torch.log1p(targ_state)) * bw
        if weights is not None:
            loss = loss * weights.view(-1, 1)
        return loss.mean()

    def total_huber(pred_cum, targ_cum, weights=None):
        loss = huber(torch.log1p(pred_cum[:, 0]), torch.log1p(targ_cum[:, 0]))
        if weights is not None:
            loss = loss * weights
        return loss.mean()

    hist = []
    for _ in range(EPOCHS):
        opt.zero_grad()
        onset_logit, cum, S = pinn(xb)
        state = cum_to_state_torch(cum)
        loss_state = state_huber(state, ys, row_w)
        loss_total = total_huber(cum, yc, row_w)
        loss = loss_state + W_TOTAL * loss_total + W_ONSET * bce(onset_logit, onset_t)

        loss_trans = torch.tensor(0.0)
        if variant.w_trans > 0 and len(cur_pos) > 0:
            if xb_self is not None:
                _, cum_self, _ = pinn(xb_self)
                prev_base = cum_self[prev_pos]
            else:
                prev_base = cum[prev_pos]
            pred_cum_trans = yc[prev_pos] + torch.relu(cum[cur_pos] - prev_base)
            pred_state_trans = cum_to_state_torch(pred_cum_trans)
            loss_trans = state_huber(pred_state_trans, ys[cur_pos], row_w[cur_pos])
            loss = loss + variant.w_trans * loss_trans

        loss_traj = torch.tensor(0.0)
        if variant.w_traj > 0 and traj_data is not None:
            _, cum_future, _ = pinn(traj_xf)
            _, cum_anchor, _ = pinn(traj_xa)
            pred_cum_traj = traj_anchor_yc + torch.relu(cum_future - cum_anchor)
            pred_state_traj = cum_to_state_torch(pred_cum_traj)
            loss_traj = state_huber(pred_state_traj, traj_ys, traj_row_w)
            loss_traj = loss_traj + W_TOTAL * total_huber(pred_cum_traj, traj_yc, traj_row_w)
            loss = loss + variant.w_traj * loss_traj

        loss_phys = torch.tensor(0.0)
        if variant.w_phys > 0:
            W = tm.integrated_svecd(pinn, er, dn, order, inv, prev_idx)
            ls = torch.log(S + 1e-6)
            lw = torch.log(W + 1e-9)
            loss_phys = torch.mean(((ls - ls.mean()) - (lw - lw.mean())) ** 2)
            loss = loss + variant.w_phys * loss_phys

        loss_mono = torch.tensor(0.0)
        if W_MONO > 0:
            g = torch.autograd.grad(S.sum(), xb, create_graph=True)[0]
            loss_mono = torch.relu(-g[:, tm.I_AGE]).mean() + torch.relu(-g[:, tm.I_N]).mean()
            loss = loss + W_MONO * loss_mono

        loss_alpha = torch.tensor(0.0)
        if W_ALPHA > 0 and alpha_prior is not None and np.isfinite(alpha_prior):
            loss_alpha = (pinn.alpha - alpha_prior) ** 2
            loss = loss + W_ALPHA * loss_alpha

        loss.backward()
        opt.step()
        sched.step()
        hist.append([
            float(loss_state),
            float(loss_total),
            float(loss_trans),
            float(loss_traj),
            float(loss_phys),
            float(loss_mono),
            float(loss_alpha),
        ])
    return pinn, np.asarray(hist)


def predict_absolute(pinn, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pinn.eval()
    with torch.no_grad():
        _, cum, _ = pinn(torch.tensor(Xte))
    cum_np = enforce_order(cum.numpy())
    return cum_to_state_np(cum_np), cum_np


def predict_transition(pinn, Xte: np.ndarray, te_global: np.ndarray, Xall: np.ndarray, imp, scl) -> tuple[np.ndarray, np.ndarray]:
    pred_state, pred_cum = predict_absolute(pinn, Xte)
    out_cum = pred_cum.copy()
    has_prev = np.array([PREV[i] >= 0 for i in te_global])
    if has_prev.any():
        prev_idx = np.array([PREV[i] for i in te_global[has_prev]])
        Xprev_source = self_anchor_matrix() if SELF_ANCHOR_BASELINE else Xall
        Xprev = transform_indices(Xprev_source, imp, scl, prev_idx)
        _, pred_prev_cum = predict_absolute(pinn, Xprev)
        out_cum[has_prev] = Y_CUM[prev_idx] + np.maximum(0.0, pred_cum[has_prev] - pred_prev_cum)
    out_cum = enforce_order(np.clip(out_cum, 0, 100))
    return cum_to_state_np(out_cum), out_cum


def metrics(name: str, pred_state: np.ndarray, pred_cum: np.ndarray, te: np.ndarray, split: str, variant: Variant) -> dict:
    y_state = Y_STATE[te]
    y_cum = Y_CUM[te]
    y_total = y_state.sum(axis=1)
    p_total = np.clip(pred_state.sum(axis=1), 0, 100)
    crk = y_total > 0
    row = {
        "split": split,
        "variant": name,
        "w_phys": variant.w_phys,
        "w_trans": variant.w_trans,
        "w_traj": variant.w_traj,
        "band_w": ",".join(f"{x:g}" for x in STATE_BAND_W.tolist()),
        "w_total": W_TOTAL,
        "w_onset": W_ONSET,
        "w_cracked": W_CRACKED,
        "huber_delta": HUBER_DELTA,
        "seed": variant.seed,
        "RMSE_total": float(np.sqrt(mean_squared_error(y_total, p_total))),
        "MAE_total": float(mean_absolute_error(y_total, p_total)),
        "R2_total": float(r2_score(y_total, p_total)),
        "R2log_total": float(r2_score(np.log1p(y_total), np.log1p(p_total))),
        "OnsetAUC": float(roc_auc_score(crk, p_total)) if len(np.unique(crk)) > 1 else np.nan,
    }
    for j, lab in enumerate(["LOW_ONLY", "MOD_ONLY", "HIGH_ONLY"]):
        row[f"R2log_{lab}"] = float(r2_score(np.log1p(y_state[:, j]), np.log1p(np.clip(pred_state[:, j], 0, 100))))
    for j, lab in enumerate(["CUM_L", "CUM_M", "CUM_H"]):
        row[f"R2log_{lab}"] = float(r2_score(np.log1p(y_cum[:, j]), np.log1p(np.clip(pred_cum[:, j], 0, 100))))
    true_mean_class = np.divide(
        y_state @ np.array([1.0, 2.0, 3.0]),
        y_total,
        out=np.zeros(len(y_total), dtype=float),
        where=y_total > 0,
    )
    pred_mean_class = np.divide(
        pred_state @ np.array([1.0, 2.0, 3.0]),
        p_total,
        out=np.zeros(len(p_total), dtype=float),
        where=p_total > 0,
    )
    cracked = y_total > 0
    row["MAE_mean_class_cracked"] = float(mean_absolute_error(true_mean_class[cracked], pred_mean_class[cracked])) if cracked.any() else np.nan
    return row


def run_temporal_sweep() -> tuple[pd.DataFrame, Variant]:
    tr, te = temporal_split()
    Xtr, Xte, Xall, imp, scl = scaled_matrices(tr, te, anchor_on=True)
    Xtr_self = transform_indices(self_anchor_matrix(Xall), imp, scl, tr) if SELF_ANCHOR_BASELINE else None
    traj_data = make_trajectory_data(tr, imp, scl, Xall) if any(w > 0 for w in W_TRAJ_LIST) else None
    if traj_data is not None:
        print(f"Trajectory pairs ({TRAJ_PAIR_MODE}): {traj_data['n_pairs']}", flush=True)
    variants = []
    for w_phys in W_PHYS_LIST:
        for w_traj in W_TRAJ_LIST:
            variants.append(Variant(f"state_absolute_w{w_phys:g}_traj{w_traj:g}", w_phys, 0.0, SEED_OFFSET, w_traj))
            for w_trans in W_TRANS_LIST:
                variants.append(
                    Variant(f"state_transition_w{w_phys:g}_t{w_trans:g}_traj{w_traj:g}", w_phys, w_trans, SEED_OFFSET, w_traj)
                )

    rows = []
    for i, var in enumerate(variants, 1):
        print(f"[state temporal {i}/{len(variants)}] {var.name}", flush=True)
        pinn, _ = train_variant(
            Xtr,
            Y_STATE[tr],
            Y_CUM[tr],
            tm.EPS_R[tr],
            tm.DN[tr],
            tm.groups[tr],
            tm.AGEv[tr],
            tr,
            var,
            Xtr_self,
            traj_data if var.w_traj > 0 else None,
        )
        pred_state_abs, pred_cum_abs = predict_absolute(pinn, Xte)
        rows.append(metrics(var.name + "_abs_pred", pred_state_abs, pred_cum_abs, te, "temporal", var))
        if var.w_trans > 0:
            pred_state_tr, pred_cum_tr = predict_transition(pinn, Xte, te, Xall, imp, scl)
            rows.append(metrics(var.name + "_transition_pred", pred_state_tr, pred_cum_tr, te, "temporal", var))
    res = pd.DataFrame(rows).sort_values("R2log_total", ascending=False).reset_index(drop=True)
    res.to_csv(OUT / "temporal_state_sweep.csv", index=False)
    best_row = res.iloc[0]
    best = Variant(
        str(best_row["variant"]).replace("_transition_pred", "").replace("_abs_pred", ""),
        float(best_row["w_phys"]),
        float(best_row["w_trans"]),
        0,
        float(best_row.get("w_traj", 0.0)),
    )
    return res, best


def run_scenario_a(best: Variant) -> pd.DataFrame:
    XA = tm.build_X(anchor_on=False)
    gkf = GroupKFold(n_splits=5)
    pred_state = np.full((len(tm.df), 3), np.nan)
    pred_cum = np.full((len(tm.df), 3), np.nan)
    for fold, (tr, te) in enumerate(gkf.split(XA, Y_TOTAL, tm.groups), 1):
        print(f"[state scenario A {fold}/5]", flush=True)
        Xtr, Xte, _, _, _ = scaled_matrices(tr, te, anchor_on=False)
        var = Variant(f"{best.name}_scenarioA", best.w_phys, best.w_trans, SEED_OFFSET + fold - 1, best.w_traj)
        pinn, _ = train_variant(Xtr, Y_STATE[tr], Y_CUM[tr], tm.EPS_R[tr], tm.DN[tr], tm.groups[tr], tm.AGEv[tr], tr, var)
        pred_state[te], pred_cum[te] = predict_absolute(pinn, Xte)
    row = metrics(best.name + "_state_pinn", pred_state, pred_cum, np.arange(len(tm.df)), "unseen_sections", best)
    res = pd.DataFrame([row])
    res.to_csv(OUT / "scenario_a_state_unseen_sections.csv", index=False)
    return res


def first_indices() -> np.ndarray:
    first = np.full(len(tm.df), -1, dtype=int)
    for _, g in tm.df.reset_index().sort_values(["SECTION_ID", "AGE"]).groupby("SECTION_ID"):
        idx = g["index"].to_numpy()
        first[idx] = idx[0]
    return first


def fixed_anchor_matrix(
    anchor_idx: np.ndarray, base_matrix: np.ndarray | None = None
) -> np.ndarray:
    x = (tm.build_X(anchor_on=True) if base_matrix is None else base_matrix).copy()
    valid = np.where(anchor_idx >= 0)[0]
    if len(valid) == 0:
        return x.astype(np.float32)
    anc = anchor_idx[valid]
    col = {c: tm.FEATURES.index(c) for c in tm.FEATURES}
    for j, suffix in enumerate(["L", "M", "H"]):
        name = f"LAST_OBS_CPCT_{suffix}"
        if name in col:
            x[valid, col[name]] = Y_CUM[anc, j].astype(np.float32)
    if "AGE_AT_LAST_OBS" in col:
        x[valid, col["AGE_AT_LAST_OBS"]] = tm.AGEv[anc].astype(np.float32)
    if "DT_SINCE_LAST" in col:
        x[valid, col["DT_SINCE_LAST"]] = (tm.AGEv[valid] - tm.AGEv[anc]).astype(np.float32)
    return x.astype(np.float32)


def self_anchor_matrix(base_matrix: np.ndarray | None = None) -> np.ndarray:
    return fixed_anchor_matrix(np.arange(len(tm.df), dtype=int), base_matrix)


def predict_transition_with_anchor(pinn, eval_idx: np.ndarray, anchor_idx: np.ndarray, Xall: np.ndarray, imp, scl, Xanchor: np.ndarray | None = None):
    Xeval = transform_indices(Xall, imp, scl, eval_idx)
    pred_state_abs, pred_cum_abs = predict_absolute(pinn, Xeval)
    valid = anchor_idx >= 0
    out_cum = pred_cum_abs.copy()
    if valid.any():
        anchor_source = Xanchor if Xanchor is not None else Xall
        Xanc = transform_indices(anchor_source, imp, scl, anchor_idx[valid])
        _, pred_anchor_cum = predict_absolute(pinn, Xanc)
        out_cum[valid] = Y_CUM[anchor_idx[valid]] + np.maximum(0.0, pred_cum_abs[valid] - pred_anchor_cum)
    out_cum = enforce_order(out_cum)
    return cum_to_state_np(out_cum), out_cum


def run_warm_start(best: Variant) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=5)
    first = first_indices()
    pred_first_s = np.full((len(tm.df), 3), np.nan)
    pred_first_c = np.full((len(tm.df), 3), np.nan)
    pred_roll_s = np.full((len(tm.df), 3), np.nan)
    pred_roll_c = np.full((len(tm.df), 3), np.nan)
    eval_rows = []
    for fold, (tr, te_all) in enumerate(gkf.split(tm.build_X(anchor_on=False), Y_TOTAL, tm.groups), 1):
        eval_idx = np.array([i for i in te_all if first[i] != i], dtype=int)
        eval_rows.extend(eval_idx.tolist())
        print(f"[state warm {fold}/5] eval={len(eval_idx)}", flush=True)
        Xtr, _, Xall, imp, scl = scaled_matrices(tr, eval_idx, anchor_on=True)
        Xroll = fixed_anchor_matrix(PREV, Xall) if SELF_ANCHOR_BASELINE else Xall
        Xfirst = fixed_anchor_matrix(first, Xall)
        Xself = self_anchor_matrix(Xall) if SELF_ANCHOR_BASELINE else None
        Xtr_self = transform_indices(Xself, imp, scl, tr) if SELF_ANCHOR_BASELINE else None
        traj_data = make_trajectory_data(tr, imp, scl, Xall) if best.w_traj > 0 else None
        if traj_data is not None:
            print(f"[state warm {fold}/5] trajectory pairs={traj_data['n_pairs']}", flush=True)
        var = Variant(f"{best.name}_warm{fold}", best.w_phys, best.w_trans, SEED_OFFSET + fold - 1, best.w_traj)
        pinn, _ = train_variant(
            Xtr,
            Y_STATE[tr],
            Y_CUM[tr],
            tm.EPS_R[tr],
            tm.DN[tr],
            tm.groups[tr],
            tm.AGEv[tr],
            tr,
            var,
            Xtr_self,
            traj_data,
        )
        pred_first_s[eval_idx], pred_first_c[eval_idx] = predict_transition_with_anchor(
            pinn, eval_idx, first[eval_idx], Xfirst, imp, scl, Xself
        )
        pred_roll_s[eval_idx], pred_roll_c[eval_idx] = predict_transition_with_anchor(
            pinn, eval_idx, np.array([PREV[i] for i in eval_idx]), Xroll, imp, scl, Xself
        )
    idx = np.array(sorted(set(eval_rows)), dtype=int)
    rows = [
        metrics(best.name + "_first_anchor_state", pred_first_s[idx], pred_first_c[idx], idx, "warm_unseen_sections", best),
        metrics(best.name + "_rolling_anchor_state", pred_roll_s[idx], pred_roll_c[idx], idx, "warm_unseen_sections", best),
    ]
    for row in rows:
        row["n_eval_rows"] = int(len(idx))
        row["n_eval_sections"] = int(tm.df.loc[idx, "SECTION_ID"].nunique())
    res = pd.DataFrame(rows).sort_values("R2log_total", ascending=False).reset_index(drop=True)
    res.to_csv(OUT / "warm_start_state_unseen_sections.csv", index=False)
    return res


def write_report(temporal: pd.DataFrame, best: Variant, scen_a: pd.DataFrame, warm: pd.DataFrame) -> None:
    manifest = {
        "dataset": TAG,
        "feature_tag": os.environ.get("SVECD_FEATURE_TAG", TAG),
        "multiband_anchor": os.environ.get("SVECD_MULTIBAND_ANCHOR", "0") == "1",
        "self_anchor_baseline": SELF_ANCHOR_BASELINE,
        "split_local_dense": SPLIT_LOCAL_DENSE,
        "seed_offset": SEED_OFFSET,
        "trajectory_pair_mode": TRAJ_PAIR_MODE,
        "trajectory_max_pairs": TRAJ_MAX_PAIRS,
        "loss": {
            "band_w": STATE_BAND_W.tolist(),
            "w_total": W_TOTAL,
            "w_onset": W_ONSET,
            "w_cracked": W_CRACKED,
            "huber_delta": HUBER_DELTA,
        },
        "pure_definition": "S-VECD PINN only; trained on disjoint LTPP severity-state targets.",
        "state_targets": STATE_COLS,
        "best_variant": best.__dict__,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(OUT / "RESULTS_SEVERITY_STATE_PINN.md", "w") as f:
        f.write("# Severity-State Pure PINN Results\n\n")
        f.write("This model trains on disjoint LTPP low/moderate/high fatigue-cracking severity states, while retaining ordered cumulative thresholds internally.\n\n")
        f.write("## Temporal Sweep\n")
        f.write(temporal.head(12).to_markdown(index=False))
        f.write("\n\n## Scenario A: Cold-Start Unseen Sections\n")
        f.write(scen_a.to_markdown(index=False))
        f.write("\n\n## Warm-Start Unseen Sections\n")
        f.write(warm.to_markdown(index=False))
        f.write("\n")


def write_temporal_report(temporal: pd.DataFrame, best: Variant) -> None:
    manifest = {
        "dataset": TAG,
        "feature_tag": os.environ.get("SVECD_FEATURE_TAG", TAG),
        "run_parts": RUN_PARTS,
        "multiband_anchor": os.environ.get("SVECD_MULTIBAND_ANCHOR", "0") == "1",
        "self_anchor_baseline": SELF_ANCHOR_BASELINE,
        "split_local_dense": SPLIT_LOCAL_DENSE,
        "seed_offset": SEED_OFFSET,
        "trajectory_pair_mode": TRAJ_PAIR_MODE,
        "trajectory_max_pairs": TRAJ_MAX_PAIRS,
        "loss": {
            "band_w": STATE_BAND_W.tolist(),
            "w_total": W_TOTAL,
            "w_onset": W_ONSET,
            "w_cracked": W_CRACKED,
            "huber_delta": HUBER_DELTA,
        },
        "pure_definition": "S-VECD PINN only; trained on disjoint LTPP severity-state targets.",
        "state_targets": STATE_COLS,
        "best_variant": best.__dict__,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(OUT / "RESULTS_SEVERITY_STATE_PINN.md", "w") as f:
        f.write("# Severity-State Pure PINN Results\n\n")
        f.write("Temporal-only feature-screening run on disjoint LTPP severity states.\n\n")
        f.write("## Temporal Sweep\n")
        f.write(temporal.head(12).to_markdown(index=False))
        f.write("\n")


def main() -> None:
    print(f"Severity-state PINN output -> {OUT}")
    print(f"State targets: {STATE_COLS}")
    print(f"Run parts: {RUN_PARTS}")
    temporal, best = run_temporal_sweep()
    print("\nBest severity-state temporal:")
    print(temporal.head(1).to_string(index=False))
    if RUN_PARTS in {"temporal", "screen"}:
        write_temporal_report(temporal, best)
        print(f"\nDone temporal screen -> {OUT}")
        return
    scen_a = run_scenario_a(best)
    warm = run_warm_start(best)
    write_report(temporal, best, scen_a, warm)
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
