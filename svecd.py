"""
svecd.py
========
Phase 3: the S-VECD (simplified viscoelastic continuum damage) governing physics that replaces
Miner's law as the mechanistic core of the fatigue model.

This module is pure NumPy and serves two roles:
  (1) a deterministic forward simulator that integrates damage over a section's real axle-load
      spectrum (used to sanity-check the mechanism and to seed the dataset), and
  (2) the reference definitions of pseudo-strain, the damage increment, the C(S) characteristic
      curve, and the damage->cracking transfer, which are mirrored in PyTorch inside the PINN
      (train_models.py) so the physics loss enforces exactly these relations.

Equations
  pseudo-strain amplitude:   eps_R = (|E*|(T,f) / E_R) * eps_t
  damage increment:          dS/dN = ( 0.5 * eps_R^2 * (-dC/dS) )^(alpha/(1+alpha))
  characteristic curve:      C(S) = 1 - C1 * S^C2          (-dC/dS = C1*C2*S^(C2-1))
  spectrum accumulation:     S_{k} = S_{k-1} + Σ_{g,l} n_{g,l} * dS/dN( eps_R(g,l), S_{k-1} )
  field cracking transfer:   C_PCT = 100 * sigmoid( beta1 * (log10(S) - beta0) )

alpha ~ 1 + 1/m (m = relaxation log-log slope), typically 2.0-3.5.  The absolute level of S is
absorbed by the trainable transfer (beta0, beta1) and k-scale, so the model identifies the
*shape* of damage growth from real strain, not just an arbitrary level.

Run:  python3 "04 - Claude/Claude - Third/svecd.py"   (forward-simulation self-test)
"""

import numpy as np

# literature-typical defaults (the PINN makes these trainable)
DEFAULTS = dict(C1=0.0008, C2=0.45, alpha=2.5, beta0=0.5, beta1=2.0)
E_R_DEFAULT_MPA = None    # set per-run to |E*|(20C,10Hz); pseudo-strain is ratio-scaled


# --------------------------------------------------------------------------------------
# Core relations (mirror these exactly in torch)
# --------------------------------------------------------------------------------------
def pseudo_strain(eps_t_micro, estar_tf_mpa, E_R_mpa):
    """Pseudo-strain amplitude (dimensionless) from LEA tensile microstrain and |E*| ratio."""
    eps_t = np.asarray(eps_t_micro, dtype=float) * 1e-6
    return (np.asarray(estar_tf_mpa, dtype=float) / E_R_mpa) * eps_t


def cs_curve(S, C1, C2):
    """Pseudo-stiffness ratio C(S) = 1 - C1*S^C2 (clamped to (0,1])."""
    S = np.maximum(np.asarray(S, dtype=float), 0.0)
    return np.clip(1.0 - C1 * np.power(S + 1e-12, C2), 1e-6, 1.0)


def neg_dC_dS(S, C1, C2):
    """-dC/dS = C1*C2*S^(C2-1) >= 0."""
    S = np.maximum(np.asarray(S, dtype=float), 1e-12)
    return C1 * C2 * np.power(S, C2 - 1.0)


def dS_dN(eps_R, S, C1, C2, alpha):
    """Per-cycle damage rate (S-VECD)."""
    drive = 0.5 * np.square(eps_R) * neg_dC_dS(S, C1, C2)
    return np.power(np.maximum(drive, 0.0), alpha / (1.0 + alpha))


def transfer(S, beta0, beta1):
    """Map damage S -> field cracking percent via a sigmoid on log10(S)."""
    S = np.maximum(np.asarray(S, dtype=float), 1e-9)
    return 100.0 / (1.0 + np.exp(-beta1 * (np.log10(S) - beta0)))


def transfer_multiband(S, C1, C2, beta1, beta0_L, beta0_M, beta0_H, A_max=100.0):
    """Structural damage->severity transfer for the multi-head model (reference; mirrored in torch).

    The single damage scalar S drives three shifted sigmoids on the damage axis
        g = log10(1 - C(S)) = log10(C1 * S^C2)
    with ORDERED onset thresholds beta0_L <= beta0_M <= beta0_H and a shared area ceiling A_max:
        C_PCT_b = A_max * sigmoid(beta1 * (g - beta0_b)),  b in {L, M, H}.
    Because the thresholds are ordered and the sigmoid is monotone, this guarantees
        C_PCT_L >= C_PCT_M >= C_PCT_H  for every S  (cumulative severity-or-greater cracking).
    Returns (crack_L, crack_M, crack_H), each in [0, A_max].
    """
    S = np.maximum(np.asarray(S, dtype=float), 0.0)
    g = np.log10(np.clip(C1 * np.power(S + 1e-12, C2), 1e-6, 1.0))
    sig = lambda b0: A_max / (1.0 + np.exp(-beta1 * (g - b0)))
    return sig(beta0_L), sig(beta0_M), sig(beta0_H)


# --------------------------------------------------------------------------------------
# Deterministic forward simulator over an axle-load spectrum
# --------------------------------------------------------------------------------------
def accumulate_damage(intervals, params=None, S0=1e-3):
    """
    Integrate S over a section's survey intervals.

    `intervals` : list of dicts, one per survey interval, each:
        {"eps_R": array of pseudo-strain amplitudes per (axle-group, load-level),
         "counts": array of matching cycle counts n_{g,l}}
    Returns array S after each interval (cumulative).
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    S = float(S0)
    out = []
    for it in intervals:
        epsR = np.atleast_1d(np.asarray(it["eps_R"], dtype=float))
        n = np.atleast_1d(np.asarray(it["counts"], dtype=float))
        # increment from each spectrum bin at the interval-start state (explicit Euler)
        dS = np.sum(n * dS_dN(epsR, S, p["C1"], p["C2"], p["alpha"]))
        S = S + float(dS)
        out.append(S)
    return np.array(out), p


# --------------------------------------------------------------------------------------
# Self-test: simulate synthetic sections and check the mechanism is well behaved
# --------------------------------------------------------------------------------------
def _selftest():
    print("Phase 3 self-test: S-VECD forward simulation on synthetic sections")
    E_R = 8000.0   # MPa reference |E*|
    # three sections: thin/soft (high strain) vs thick/stiff (low strain)
    cases = {
        "thin-soft  (eps_t~280ue)": 280.0,
        "medium     (eps_t~170ue)": 170.0,
        "thick-stiff(eps_t~ 90ue)": 90.0,
    }
    # 20 yearly intervals, ~1e6 axles/yr split across 3 axle-load levels
    counts = np.array([6e5, 3e5, 1e5])
    load_factor = np.array([0.8, 1.0, 1.4])     # light / standard / heavy -> scales strain
    for label, et in cases.items():
        intervals = []
        for _ in range(20):
            epsR = pseudo_strain(et * load_factor, E_R, E_R)   # obs at ~ref temp
            intervals.append({"eps_R": epsR, "counts": counts})
        S, p = accumulate_damage(intervals)
        C = cs_curve(S, p["C1"], p["C2"])
        crack = transfer(S, p["beta0"], p["beta1"])
        mono_S = np.all(np.diff(S) >= 0)
        mono_C = np.all(np.diff(C) <= 1e-12)
        print(f"  {label}: S_final={S[-1]:.3g}  C_final={C[-1]:.3f}  "
              f"crack_final={crack[-1]:5.1f}%  (S monotone up={mono_S}, C monotone down={mono_C})")
        assert mono_S and mono_C

    # ordering: higher strain -> more cracking
    cr = [transfer(accumulate_damage(
            [{"eps_R": pseudo_strain(et * load_factor, E_R, E_R), "counts": counts}
             for _ in range(20)])[0], DEFAULTS["beta0"], DEFAULTS["beta1"])[-1]
          for et in (280.0, 170.0, 90.0)]
    print(f"  crack% by strain (280/170/90 ue): {cr[0]:.1f} / {cr[1]:.1f} / {cr[2]:.1f}")
    assert cr[0] > cr[1] > cr[2], "higher strain must yield more cracking"

    # multi-band structural transfer: ordering L>=M>=H for every S, all monotone up in S
    Sgrid = np.logspace(-2, 3, 200)
    cL, cM, cH = transfer_multiband(Sgrid, C1=0.05, C2=0.45, beta1=2.0,
                                    beta0_L=-1.0, beta0_M=-0.3, beta0_H=0.4, A_max=80.0)
    assert np.all(cL >= cM - 1e-9) and np.all(cM >= cH - 1e-9), "severity bands must be ordered L>=M>=H"
    assert np.all(np.diff(cL) >= -1e-9) and np.all(np.diff(cH) >= -1e-9), "bands must rise with S"
    print(f"  multiband transfer: ordering L>=M>=H OK; "
          f"at S={Sgrid[-1]:.0f} -> L={cL[-1]:.1f} M={cM[-1]:.1f} H={cH[-1]:.1f}%")
    print("Phase 3 OK (S monotone up, C monotone down, crack% increases with strain, bands ordered).")


if __name__ == "__main__":
    _selftest()
