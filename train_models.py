"""
train_models.py  (Claude - Third)
=================================
Phase 5: GPS-1 + SPS-1 fatigue-cracking models with S-VECD physics on deterministic LEA strain.

The PINN replaces the Miner snapshot with an S-VECD damage-evolution core:
  * shared trunk -> monotone latent damage S >= 0
  * physics loss ties S to the time-integrated S-VECD damage built from the REAL pseudo-strain
    EPS_R (from LEA |E*| ratio) and REAL axle counts N_CYCLES (NALS):
        W(t_k) = Σ_{j<=k} dN_j * (0.5 * EPS_R_j^2)^(alpha/(1+alpha))
    (alpha trainable; a log k-scale sets the absolute level — only the *shape* from real strain
     and traffic can satisfy the residual).
  * characteristic curve  C(S) = 1 - C1*S^C2  (C1>0, C2>0 trainable)  -> damage fraction (1-C)
  * transfer  C_PCT = 100*sigmoid(beta1*(log10(1-C) - beta0)).
  * monotonicity loss: dS/dN >= 0, dS/dAGE >= 0 (autograd).

Because the target is ~61% zeros, we also fit a HURDLE model: an onset classifier
(XGBoost) x a severity regressor, and report onset AUC alongside R2.

Validation (unchanged from Claude-Second):
  A. Unseen sections  - GroupKFold(5) by SECTION_ID.
  B. Temporal extrapolation - hold out each progression section's latest survey.

Run:  python3 "04 - Claude/Claude - Third/train_models.py"   ->  outputs_svecd/
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score)
import xgboost as xgb

warnings.simplefilter("ignore")
torch.manual_seed(0)
np.random.seed(0)

PROJ = os.path.dirname(os.path.abspath(__file__))
EXP_FILTER = os.environ.get("SVECD_EXP_FILTER")          # e.g. "GPS1" to subset one experiment
TAG = os.environ.get("SVECD_TAG", "")                    # dataset tag (e.g. "_expanded")
FEATURE_TAG = os.environ.get("SVECD_FEATURE_TAG", TAG)   # optional feature-list tag for ablations
OUT_SUFFIX = os.environ.get("SVECD_OUT_SUFFIX", "")       # output-only dir suffix (compare runs)
HURDLE = os.environ.get("SVECD_HURDLE", "0") == "1"       # explicit two-stage PINN (ablation)
_suffix = (TAG + (f"_{EXP_FILTER}" if EXP_FILTER else "")
           + ("_hurdle" if HURDLE else "") + OUT_SUFFIX)
OUTDIR = os.path.join(PROJ, "outputs_svecd" + _suffix)
os.makedirs(OUTDIR, exist_ok=True)

with open(os.path.join(PROJ, f"feature_cols{FEATURE_TAG}.txt")) as f:
    FEATURES = [l.strip() for l in f if l.strip()]

# Multi-head severity mode: predict cumulative low/mod/high cracking with one ordered S-VECD
# transfer.  Auto-enabled when the dataset carries the severity columns (or forced via env).
df = pd.read_csv(os.path.join(PROJ, f"fatigue_svecd{TAG}.csv"), parse_dates=["SURVEY_DATE", "ORIGIN"])
SEV_COLS = ["C_PCT_L", "C_PCT_M", "C_PCT_H"]
MULTIHEAD = (os.environ.get("SVECD_MULTIHEAD", "auto") != "0") and all(c in df.columns for c in SEV_COLS)
TARGETS = SEV_COLS if MULTIHEAD else ["C_PCT"]
PRIMARY = TARGETS[0]                                  # band used for onset / dropna / diagnostics
N_BANDS = len(TARGETS)
# band weights (env-tunable). Default favors the headline L band (total cracking, the thesis
# metric); M/H get less since they are sparser/noisier and the cumulative target already couples them.
_BW_DEFAULT = "1.0,0.6,0.4"
BAND_W = (np.array([float(x) for x in os.environ.get("SVECD_BAND_W", _BW_DEFAULT).split(",")][:N_BANDS],
                   dtype=np.float32) if MULTIHEAD else np.array([1.0], np.float32))
print(f"Features: {FEATURES}")
if FEATURE_TAG != TAG:
    print(f"Feature-list tag: {FEATURE_TAG} for dataset tag: {TAG}")
print(f"Mode: {'MULTI-HEAD severity ' + str(TARGETS) if MULTIHEAD else 'single-head C_PCT'}")

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
df = df.dropna(subset=[PRIMARY, "AGE", "N_CYCLES", "EPS_T_MICRO", "EPS_R"]).reset_index(drop=True)
if EXP_FILTER:
    df = df[df["EXPERIMENT"] == EXP_FILTER].reset_index(drop=True)
    print(f"[experiment filter] {EXP_FILTER} only")
df = df.sort_values(["SECTION_ID", "AGE"]).reset_index(drop=True)
print(f"Model-ready rows: {len(df)} | sections: {df.SECTION_ID.nunique()} | "
      f"zero-{PRIMARY}%: {100*(df[PRIMARY]==0).mean():.1f}%")

# per-section increments of real axle count
df["DN"] = df.groupby("SECTION_ID")["N_CYCLES"].diff()
df["DN"] = df["DN"].fillna(df["N_CYCLES"]).clip(lower=0.0)

I_AGE = FEATURES.index("AGE")
I_N = FEATURES.index("N_CYCLES") if "N_CYCLES" in FEATURES else FEATURES.index("LOG_N_CYCLES")

groups = df["SECTION_ID"].values
Y = df[TARGETS].values.astype(np.float32)               # (N, n_bands) cumulative severity targets
y = Y[:, 0]                                              # primary band (C_PCT_L) for onset/diagnostics

# ---- Phase E2: section-anchoring features (temporal-R2 lever) ----
# Each row gets its section's most recent PRIOR survey cracking + the time gap, so the model can
# anchor the S-VECD trajectory to the section's observed onset and extrapolate forward.  These use
# ONLY past surveys (shift within section, sorted by AGE), so they are leakage-safe in Scenario B
# (predict a section's latest survey from its earlier ones) but MUST be neutralized in Scenario A
# (unseen sections have no history) — handled by build_X(anchor_on=False).
ANCHOR = os.environ.get("SVECD_ANCHOR", "0") == "1"
MULTIBAND_ANCHOR = os.environ.get("SVECD_MULTIBAND_ANCHOR", "0") == "1"
ANCHOR_COLS = []
if ANCHOR:
    df["LAST_OBS_CPCT_L"] = df.groupby("SECTION_ID")[PRIMARY].shift(1)
    ANCHOR_COLS = ["LAST_OBS_CPCT_L"]
    if MULTIBAND_ANCHOR and MULTIHEAD:
        for band in TARGETS[1:]:
            suffix = band.replace("C_PCT_", "")
            col = f"LAST_OBS_CPCT_{suffix}"
            df[col] = df.groupby("SECTION_ID")[band].shift(1)
            ANCHOR_COLS.append(col)
    df["AGE_AT_LAST_OBS"] = df.groupby("SECTION_ID")["AGE"].shift(1)
    df["DT_SINCE_LAST"] = df["AGE"] - df["AGE_AT_LAST_OBS"]
    ANCHOR_COLS += ["AGE_AT_LAST_OBS", "DT_SINCE_LAST"]
    if os.environ.get("SVECD_EXTRA_INTERVAL", "0") == "1":
        df["DN_SINCE_LAST"] = df.groupby("SECTION_ID")["N_CYCLES"].diff()
        df["DLOGN_SINCE_LAST"] = df.groupby("SECTION_ID")["LOG_N_CYCLES"].diff() if "LOG_N_CYCLES" in df.columns else np.nan
        df["DESAL_SINCE_LAST"] = df.groupby("SECTION_ID")["CUM_ESAL"].diff() if "CUM_ESAL" in df.columns else np.nan
        ANCHOR_COLS += ["DN_SINCE_LAST", "DLOGN_SINCE_LAST", "DESAL_SINCE_LAST"]
    FEATURES = FEATURES + [c for c in ANCHOR_COLS if c not in FEATURES]
    print(f"Anchoring ON: +{ANCHOR_COLS}")


def build_X(anchor_on):
    """Feature matrix; anchor columns neutralized (NaN -> train-median) when anchor_on=False."""
    Xdf = df[FEATURES].copy()
    if not anchor_on:
        for c in ANCHOR_COLS:
            Xdf[c] = np.nan
    return Xdf.values.astype(np.float32)


X = build_X(anchor_on=ANCHOR)                            # default matrix (used for KFold splitting)
EPS_R = df["EPS_R"].values.astype(np.float32)
DN = (df["DN"].values / 1e6).astype(np.float32)         # millions of axles per interval
AGEv = df["AGE"].values
ylog = np.log1p(Y)                                       # (N, n_bands)
onset = (y > 0).astype(int)
ALPHA_PRIOR_v = df["ALPHA_PRIOR"].values.astype(np.float32) if "ALPHA_PRIOR" in df.columns else None
W_ALPHA = float(os.environ.get("SVECD_W_ALPHA", "0.1"))   # weight for the alpha-prior regularizer
N_ENSEMBLE = int(os.environ.get("SVECD_ENSEMBLE", "1"))   # number of seed-averaged PINN members
HYBRID = os.environ.get("SVECD_HYBRID", "0") == "1"       # physics PINN + GBM residual correction


# --------------------------------------------------------------------------------------
# Segment cumulative-sum index (vectorized differentiable within-section integral)
# --------------------------------------------------------------------------------------
def segment_index(sections, ages):
    order = np.lexsort((ages, sections))
    inv = np.argsort(order)
    sec_sorted = sections[order]
    is_first = np.concatenate(([True], sec_sorted[1:] != sec_sorted[:-1]))
    group_id = np.cumsum(is_first) - 1
    start_pos = np.flatnonzero(is_first)
    prev_idx = start_pos[group_id] - 1
    return order, inv, prev_idx


# --------------------------------------------------------------------------------------
# S-VECD PINN
# --------------------------------------------------------------------------------------
class SvecdPINN(nn.Module):
    """S-VECD PINN with an n_bands structural transfer.

    One latent damage scalar S>=0 drives n_bands shifted sigmoids on the damage axis
    g = log10(1 - C(S)) with ORDERED onset thresholds (beta0_1 <= beta0_2 <= ...) and a shared
    trainable area ceiling A_max.  For n_bands=3 this yields cumulative low/mod/high cracking with
    C_PCT_L >= C_PCT_M >= C_PCT_H guaranteed by construction.  n_bands=1 recovers the original
    single-head transfer (with A_max now trainable instead of fixed at 100)."""

    def __init__(self, n_in, n_bands=1):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU())
        self.n_bands = n_bands
        self.trunk = nn.Sequential(blk(n_in, 64), blk(64, 64), blk(64, 32))
        self.damage_head = nn.Linear(32, 1)
        self.onset_head = nn.Linear(32, 1)                   # hurdle stage 1: P(C>0) logit
        self.log_kscale = nn.Parameter(torch.tensor(0.0))
        self.raw_alpha = nn.Parameter(torch.tensor(1.0))     # alpha = softplus -> ~1.3
        self.log_C1 = nn.Parameter(torch.tensor(-3.0))       # C1 ~ 0.05
        self.raw_C2 = nn.Parameter(torch.tensor(-0.5))       # C2 = softplus -> ~0.47
        self.b0 = nn.Parameter(torch.tensor(-1.0))           # first (lowest-severity) onset threshold
        self.raw_gaps = nn.Parameter(torch.zeros(max(n_bands - 1, 1)))  # softplus gaps -> ordering
        self.raw_beta1 = nn.Parameter(torch.tensor(0.5))
        # trainable ceiling, init at the known-good 100% (old fixed value); softplus(100)~100
        self.raw_Amax = nn.Parameter(torch.tensor(100.0))

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.raw_alpha) + 1.0   # alpha > 1

    def thresholds(self):
        """Ordered band thresholds beta0_b = b0 + cumsum(softplus(gaps)). Shape (n_bands,)."""
        gaps = torch.nn.functional.softplus(self.raw_gaps[: self.n_bands - 1])
        cum = torch.cat([torch.zeros(1), torch.cumsum(gaps, 0)]) if self.n_bands > 1 \
            else torch.zeros(1)
        return self.b0 + cum

    def transfer(self, S):
        """Damage S -> (N, n_bands) cracking percent via ordered sigmoids on log10(1-C(S))."""
        C1 = torch.exp(self.log_C1)
        C2 = torch.nn.functional.softplus(self.raw_C2)
        beta1 = torch.nn.functional.softplus(self.raw_beta1)
        A_max = torch.nn.functional.softplus(self.raw_Amax)
        dmg_frac = torch.clamp(C1 * torch.pow(S + 1e-9, C2), 1e-6, 1.0)   # 1 - C(S)
        g = torch.log10(dmg_frac).unsqueeze(-1)                          # (N,1)
        b0 = self.thresholds().unsqueeze(0)                              # (1,n_bands)
        return A_max * torch.sigmoid(beta1 * (g - b0))                   # (N,n_bands)

    def forward(self, x):
        """Returns (onset logit, severity bands (N,n_bands), latent damage S)."""
        h = self.trunk(x)
        S = torch.nn.functional.softplus(self.damage_head(h)).squeeze(-1)
        onset_logit = self.onset_head(h).squeeze(-1)
        return onset_logit, self.transfer(S), S


def integrated_svecd(pinn, eps_r, dN, order, inv, prev_idx):
    """W(t_k) = Σ_{j<=k} dN_j * (0.5 * eps_R_j^2)^(alpha/(1+alpha))  (per section)."""
    a = pinn.alpha
    drive = (0.5 * eps_r.pow(2)).clamp(min=1e-30).pow(a / (1.0 + a))
    incr = (dN * drive) * torch.exp(pinn.log_kscale)
    cs = torch.cumsum(incr[order], dim=0)
    offset = torch.where(prev_idx >= 0, cs[prev_idx.clamp(min=0)], torch.zeros_like(cs))
    return (cs - offset)[inv]


def train_pinn(Xtr, Ytr, eps_r, dN, sections, ages, w_phys, w_mono, alpha_prior=None, w_alpha=0.0,
               epochs=int(os.environ.get("SVECD_EPOCHS", "700")), lr=2e-3, seed=0):
    torch.manual_seed(seed)                                   # per-member seed (ensembling)
    pinn = SvecdPINN(Xtr.shape[1], n_bands=N_BANDS)
    opt = torch.optim.Adam(pinn.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber = nn.HuberLoss(delta=1.0)
    bce = nn.BCEWithLogitsLoss()
    bw = torch.tensor(BAND_W)
    xb = torch.tensor(Xtr, requires_grad=True)
    yb = torch.tensor(Ytr)                                # (N, n_bands)
    onset_t = torch.tensor((Ytr[:, 0] > 0).astype(np.float32))
    cracked = torch.tensor(Ytr[:, 0] > 0)
    er = torch.tensor(eps_r); dn = torch.tensor(dN)
    order, inv, prev_idx = segment_index(sections, ages)
    order = torch.tensor(order); inv = torch.tensor(inv); prev_idx = torch.tensor(prev_idx)

    def band_huber(pred, targ):                           # weighted sum of per-band Huber on log1p
        return sum(bw[b] * huber(torch.log1p(pred[:, b]), torch.log1p(targ[:, b]))
                   for b in range(N_BANDS))

    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        onset_logit, sev, S = pinn(xb)                    # sev: (N, n_bands)
        if HURDLE:
            # gated hurdle (ablation): combined P(onset)*severity optimized end-to-end on the real
            # objective (all rows), with BCE auxiliary to keep the gate a calibrated probability and
            # a cracked-only severity term so the S-VECD transfer still learns crack magnitude.
            pred = torch.sigmoid(onset_logit).unsqueeze(-1) * sev
            L_full = band_huber(pred, yb)
            L_onset = bce(onset_logit, onset_t)
            L_sev = band_huber(sev[cracked], yb[cracked]) if cracked.any() else torch.tensor(0.0)
            L_data = L_full + 0.5 * L_onset + 0.2 * L_sev
        else:
            # single-stage: the S-VECD log-S sigmoid transfer is itself a soft hurdle (better
            # calibrated on the zero-heavy distribution — see outputs_svecd_hurdle ablation).
            L_data = band_huber(sev, yb)
        loss = L_data
        L_phys = torch.tensor(0.0); L_mono = torch.tensor(0.0)
        if w_phys > 0:
            W = integrated_svecd(pinn, er, dn, order, inv, prev_idx)
            ls = torch.log(S + 1e-6); lw = torch.log(W + 1e-9)
            L_phys = torch.mean(((ls - ls.mean()) - (lw - lw.mean())) ** 2)
            loss = loss + w_phys * L_phys
        if w_mono > 0:
            g = torch.autograd.grad(S.sum(), xb, create_graph=True)[0]
            L_mono = torch.relu(-g[:, I_AGE]).mean() + torch.relu(-g[:, I_N]).mean()
            loss = loss + w_mono * L_mono
        if w_alpha > 0 and alpha_prior is not None and np.isfinite(alpha_prior):
            # pull the (global) S-VECD damage exponent toward the physics-derived prior
            L_alpha = (pinn.alpha - float(alpha_prior)) ** 2
            loss = loss + w_alpha * L_alpha
        loss.backward()
        opt.step(); sched.step()
        aux = float(L_onset) if HURDLE else 0.0
        hist.append((float(L_data), aux, float(L_phys), float(L_mono)))
    return pinn, hist


def pinn_predict(pinn, X_, return_parts=False):
    """Prediction: (N, n_bands) severity, or P(onset)*severity (hurdle ablation)."""
    pinn.eval()
    with torch.no_grad():
        onset_logit, sev, _ = pinn(torch.tensor(X_))
        p_on = torch.sigmoid(onset_logit)
        pred = ((p_on.unsqueeze(-1) * sev) if HURDLE else sev).numpy()   # (N, n_bands)
    if return_parts:
        return pred, p_on.numpy(), sev.numpy()
    return pred


CALMODE = os.environ.get("SVECD_CALMODE", "incr")             # "incr" (anchor+growth, best) | "resid" (log-offset)


def local_calibrate(pred_tr, pred_te, tr, te):
    """MEPDG-style per-section local calibration for temporal extrapolation.

    Two modes (SVECD_CALMODE):
      "resid" — add the log1p residual at the section's latest TRAINING survey (multiplicative level
                fix); physics sets the growth shape.
      "incr"  — anchor at the last observed cracking and add only the physics-predicted INCREMENT
                from the last training survey to now: C_now = C_last + relu(phys_now - phys_last).
                Leans on physics only for the (smaller) growth, so physics error contributes less.
    Both use only training observations (leakage-safe) and a monotone floor (cracking can't decrease)."""
    g_tr, a_tr = groups[tr], AGEv[tr]
    order = np.argsort(a_tr)                                   # latest training survey per section = last in age order
    last_resid, last_obs, last_phys = {}, {}, {}
    for i in order:
        sec = g_tr[i]
        last_resid[sec] = np.log1p(Y[tr][i]) - np.log1p(np.clip(pred_tr[i], 0, None))
        last_obs[sec] = Y[tr][i]                               # last observed cracking per band (overwrite -> highest age)
        last_phys[sec] = np.clip(pred_tr[i], 0, None)         # physics prediction at that survey
    g_te = groups[te]
    out = pred_te.copy()
    for j in range(len(te)):
        sec = g_te[j]
        if sec not in last_obs:
            continue
        if CALMODE == "incr":
            growth = np.maximum(0.0, np.clip(out[j], 0, None) - last_phys[sec])
            out[j] = last_obs[sec] + growth                   # anchor + physics-predicted increment
        else:
            out[j] = np.expm1(np.log1p(np.clip(out[j], 0, None)) + last_resid[sec])
            out[j] = np.maximum(out[j], last_obs[sec])        # monotone floor
    return _enforce_order(np.clip(out, 0, 100))


# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------
_PN = ["Hurdle S-VECD PINN", "Hurdle PINN (-phys)", "Hurdle NN (-phys-mono)"] if HURDLE \
    else ["S-VECD PINN (full)", "S-VECD PINN (-phys)", "Plain NN"]
W_PHYS_MAIN = float(os.environ.get("SVECD_W_PHYS", "0.1"))    # 0.1 from Phase-E re-sweep (with local calibration)
CONFIGS = {
    _PN[0]: dict(kind="pinn", w_phys=W_PHYS_MAIN, w_mono=0.1, w_alpha=W_ALPHA),
    _PN[1]: dict(kind="pinn", w_phys=0.0, w_mono=0.1, w_alpha=W_ALPHA),
    _PN[2]: dict(kind="pinn", w_phys=0.0, w_mono=0.0),
    "Random Forest":       dict(kind="rf"),
    "XGBoost":             dict(kind="xgb"),
    "Hurdle XGB":          dict(kind="hurdle"),
}
PINN_MAIN = _PN[0]


def fit_predict(tr, te, want_hist=False, Xmat=None, localcal=False):
    Xm = X if Xmat is None else Xmat
    imp = SimpleImputer(strategy="median").fit(Xm[tr])
    scl = StandardScaler().fit(imp.transform(Xm[tr]))
    Xtr = scl.transform(imp.transform(Xm[tr])).astype(np.float32)
    Xte = scl.transform(imp.transform(Xm[te])).astype(np.float32)
    preds, hist0 = {}, None
    for name, cfg in CONFIGS.items():
        if cfg["kind"] == "pinn":
            aprior = (float(np.nanmedian(ALPHA_PRIOR_v[tr]))
                      if (ALPHA_PRIOR_v is not None and cfg.get("w_alpha", 0) > 0) else None)
            # seed ensemble: average raw physics predictions across members (variance reduction).
            te_acc = np.zeros((len(te), N_BANDS)); tr_acc = np.zeros((len(tr), N_BANDS)); h = None
            for s in range(N_ENSEMBLE):
                pinn, hs = train_pinn(Xtr, Y[tr], EPS_R[tr], DN[tr], groups[tr], AGEv[tr],
                                      cfg["w_phys"], cfg["w_mono"],
                                      alpha_prior=aprior, w_alpha=cfg.get("w_alpha", 0.0), seed=s)
                te_acc += pinn_predict(pinn, Xte).clip(0, 100)
                tr_acc += pinn_predict(pinn, Xtr).clip(0, 100)
                if s == 0:
                    h = hs
            phys_te = te_acc / N_ENSEMBLE; phys_tr = tr_acc / N_ENSEMBLE
            if HYBRID and localcal:
                # physics + ML-residual hybrid: a small GBM explains the growth the rigid S-VECD
                # term misses (per band), on top of the mechanistic PINN backbone.  Gated to the
                # temporal setting (localcal) where each section's history feeds the residual; on
                # unseen sections (Scenario A) the residual has no anchor signal and only adds noise.
                for b in range(N_BANDS):
                    gbm = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                           random_state=0).fit(Xtr, Y[tr][:, b] - phys_tr[:, b])
                    phys_te[:, b] = (phys_te[:, b] + gbm.predict(Xte)).clip(0, 100)
                    phys_tr[:, b] = (phys_tr[:, b] + gbm.predict(Xtr)).clip(0, 100)
            pred_te = phys_te
            if localcal:
                pred_te = local_calibrate(phys_tr, pred_te, tr, te)
            preds[name] = _enforce_order(pred_te)
            if want_hist and name == PINN_MAIN:
                hist0 = h
        elif cfg["kind"] in ("rf", "xgb"):
            cols = []
            for b in range(N_BANDS):                                  # one regressor per band
                if cfg["kind"] == "rf":
                    m = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                              random_state=0, n_jobs=-1).fit(Xtr, ylog[tr][:, b])
                else:
                    m = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                         random_state=0).fit(Xtr, ylog[tr][:, b])
                cols.append(np.expm1(m.predict(Xte)))
            preds[name] = _enforce_order(np.column_stack(cols)).clip(0, 100)
        else:  # hurdle: P(onset) * E[severity | onset], per band
            clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                    random_state=0, eval_metric="logloss")
            clf.fit(Xtr, onset[tr])
            p_on = clf.predict_proba(Xte)[:, 1]
            cracked = onset[tr] == 1
            cols = []
            for b in range(N_BANDS):
                if cracked.sum() >= 20:
                    reg = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                           random_state=0).fit(Xtr[cracked], ylog[tr][cracked, b])
                    sev = np.expm1(reg.predict(Xte))
                else:
                    sev = np.full(len(te), np.expm1(ylog[tr][cracked, b].mean()) if cracked.any() else 0.0)
                cols.append(p_on * sev)
            preds[name] = _enforce_order(np.column_stack(cols)).clip(0, 100)
    return preds, hist0


def _enforce_order(P):
    """Make independent per-band baseline predictions respect L>=M>=H (cumulative monotone)."""
    if P.shape[1] <= 1:
        return P
    return np.minimum.accumulate(P, axis=1)


def mono_rate(pred):
    pred = np.asarray(pred)
    if pred.ndim > 1:
        pred = pred[:, 0]
    ok = tot = 0
    for _, g in df.assign(P=pred).groupby("SECTION_ID"):
        d = np.diff(g.sort_values("AGE")["P"].values)
        ok += (d >= -1e-6).sum(); tot += len(d)
    return ok / max(tot, 1)


def band_metrics(Ytrue, Pred):
    """Per-band R2_log for a (N,n_bands) target/prediction pair."""
    out = {}
    for b, name in enumerate(TARGETS):
        tag = name.replace("C_PCT", "R2log") if MULTIHEAD else "R2log"
        out[tag] = (r2_score(np.log1p(Ytrue[:, b]), np.log1p(Pred[:, b]))
                    if len(Ytrue) > 2 else np.nan)
    return out


def fmt(dfx):
    return dfx.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main():
    # ---- Scenario A: unseen sections ----
    print("\nScenario A: GroupKFold by section ...")
    XA = build_X(anchor_on=False)                        # unseen sections have no own-history anchor
    gkf = GroupKFold(n_splits=5)
    oof = {name: np.full((len(df), N_BANDS), np.nan) for name in CONFIGS}
    first_hist = None
    for fold, (tr, te) in enumerate(gkf.split(XA, y, groups)):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        preds, h = fit_predict(tr, te, want_hist=(fold == 0), Xmat=XA)
        for name in CONFIGS:
            oof[name][te] = preds[name]
        if fold == 0:
            first_hist = h
        print(f"  fold {fold+1}/5 done")

    crk = y > 0                                          # primary-band onset
    rowsA = []
    for name in CONFIGS:
        P = oof[name]; p = P[:, 0]                        # primary band for headline metrics
        auc = roc_auc_score(crk, p) if len(np.unique(crk)) > 1 else np.nan
        row = dict(Model=name, RMSE=np.sqrt(mean_squared_error(y, p)),
                   MAE=mean_absolute_error(y, p), R2=r2_score(y, p),
                   R2_log=r2_score(np.log1p(y), np.log1p(p)),
                   R2_crk=r2_score(y[crk], p[crk]) if crk.sum() > 2 else np.nan,
                   OnsetAUC=auc, MonoRate=mono_rate(p))
        if MULTIHEAD:
            row.update(band_metrics(Y, P))
        rowsA.append(row)
    resA = pd.DataFrame(rowsA).sort_values("R2_log", ascending=False).reset_index(drop=True)
    resA.to_csv(os.path.join(OUTDIR, "model_comparison.csv"), index=False)
    print("\n=== SCENARIO A — unseen sections (GroupKFold) ===")
    print(fmt(resA))

    # ---- Scenario B: temporal extrapolation ----
    print("\nScenario B: temporal extrapolation ...")
    last_idx = (df[df.IS_PROGRESSION == 1].sort_values("AGE")
                  .groupby("SECTION_ID").tail(1).index.values)
    te_mask = np.zeros(len(df), bool); te_mask[last_idx] = True
    trB, teB = np.where(~te_mask)[0], np.where(te_mask)[0]
    XB = build_X(anchor_on=ANCHOR)                        # Scenario B may use each section's own history
    predB, _ = fit_predict(trB, teB, Xmat=XB, localcal=ANCHOR)
    YB = Y[teB]; yB = YB[:, 0]; crkB = yB > 0
    rowsB = []
    for name in CONFIGS:
        P = predB[name]; p = P[:, 0]
        auc = roc_auc_score(crkB, p) if len(np.unique(crkB)) > 1 else np.nan
        row = dict(Model=name, RMSE=np.sqrt(mean_squared_error(yB, p)),
                   MAE=mean_absolute_error(yB, p), R2=r2_score(yB, p),
                   R2_log=r2_score(np.log1p(yB), np.log1p(p)), OnsetAUC=auc)
        if MULTIHEAD:
            row.update(band_metrics(YB, P))
        rowsB.append(row)
    resB = pd.DataFrame(rowsB).sort_values("R2_log", ascending=False).reset_index(drop=True)
    resB.to_csv(os.path.join(OUTDIR, "model_comparison_temporal.csv"), index=False)
    print(f"\n=== SCENARIO B — temporal extrapolation ({len(teB)} latest surveys) ===")
    print(fmt(resB))

    _plots(oof, predB, yB, first_hist)
    _results_md(resA, resB, len(teB))
    print("\nAll outputs ->", OUTDIR)


def _plots(oof, predB, yB, first_hist):
    if first_hist:
        h = np.array(first_hist)
        plt.figure(figsize=(7, 4))
        plt.plot(h[:, 0], label="data (Huber)")
        if HURDLE:
            plt.plot(h[:, 1], label="onset (BCE)")
        plt.plot(h[:, 2], label="S-VECD (integrated)")
        plt.plot(h[:, 3], label="monotonicity")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.yscale("log")
        plt.title(f"{PINN_MAIN} training losses — fold 1"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "pinn_learning_curve.png"), dpi=120); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, name in zip(axes.ravel(), CONFIGS):
        p = oof[name][:, 0]                              # primary band (C_PCT_L)
        ax.scatter(y, p, s=8, alpha=0.4); ax.plot([0, 100], [0, 100], "r--", lw=1)
        ax.set_title(f"{name}  (R²={r2_score(y, p):.2f})")
        ax.set_xlabel(f"actual {PRIMARY}%"); ax.set_ylabel("pred %"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    fig.suptitle(f"Scenario A — predicted vs actual {PRIMARY} (out-of-fold)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "pred_vs_actual.png"), dpi=120); plt.close()


def _results_md(resA, resB, nB):
    with open(os.path.join(OUTDIR, "RESULTS.md"), "w") as f:
        tgt = f"multi-head severity {TARGETS}" if MULTIHEAD else "C_PCT"
        f.write(f"""# Results — GPS-1 + SPS-1 Fatigue: S-VECD PINN on LEA strain (SDR39)

{len(df)} rows / {df.SECTION_ID.nunique()} sections; target {tgt} (% lane area), primary-band
{100*(y==0).mean():.0f}% zeros.  {'Cumulative bands: C_PCT_L>=C_PCT_M>=C_PCT_H, one ordered S-VECD transfer.' if MULTIHEAD else ''}
Physics = **S-VECD damage evolution** on deterministic LEA pseudo-strain (EPS_R) and real NALS
axle counts (N_CYCLES).  Hurdle model handles zero-inflation (onset classifier x severity).

## Scenario A — generalize to UNSEEN sections (GroupKFold-5)
{fmt(resA)}

## Scenario B — TEMPORAL EXTRAPOLATION to latest survey ({nB} points)
{fmt(resB)}

## Read
- OnsetAUC is the honest headline given {100*(y==0).mean():.0f}% zeros.
- Claude-Second (Miner proxy) baselines: A R²(log)=0.04, B R²(log)=0.48.
- Features: {FEATURES}
""")


if __name__ == "__main__":
    main()
