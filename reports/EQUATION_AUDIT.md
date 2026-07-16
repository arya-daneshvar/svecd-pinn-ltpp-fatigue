# Equation Audit: Severity-Aware S-VECD-Informed PINN

## Audit conclusion and scope

The repository does **not** contain one internally consistent object that is simultaneously the newest “best pure PINN,” the frozen reproducibility model, and the model described in the current Methods outline.

Two candidates exist:

1. **Operational frozen paper model (primary subject of this audit).** The paper-facing runner, frozen configuration, model card, Results draft, mainstream comparisons, limited-data work, and July 2026 final-validation code all use `04 - Agent Codes/Codex - Frozen`. Its code, dataset, and runner are byte-identical to the corresponding files in `Codex - Frozen GitHub`. It has a `64,64,32` trunk, unbounded-above `alpha = 1 + softplus(raw_alpha)`, at most 20,000 trajectory pairs (the cap is inactive), and the selected seed-0 temporal `R2log_total = 0.7887469`.
2. **Later best-kept development candidate.** `Codex - Second/EXPERIMENT_MEMORY.md` calls a later `128,128,64`, `alpha in (2.0,3.5)`, uncapped-all-pairs configuration the “Current Best Kept Setup.” Its seed-0 temporal `R2log_total = 0.7980059`. It was produced through the trajectory-sampling sweep, but it was not copied into the frozen runner and is not what `Codex - FinalValidation` imports.

For a public reproducibility repository, “final” must mean the model actually used to generate the reported and final-validated results. Therefore Sections A–D below specify the **operational frozen paper model** exactly. Section E records the later-candidate conflict and the decision that must be made before publication.

No source file was modified during this audit.

## 1. Verified execution path

The operational frozen execution path is:

1. `Codex - Frozen/run_frozen_best.py:28-57`, `main()`, writes environment variables and launches `train_severity_state_pinn.py` in the frozen directory.
2. `Codex - Frozen/train_severity_state_pinn.py:677-695`, `main()`, calls `run_temporal_sweep()` and, for `--parts all`, `run_scenario_a()` and `run_warm_start()`.
3. `run_temporal_sweep()` at lines 450-497 constructs the temporal split and calls `train_variant()` at lines 249-380.
4. `train_variant()` instantiates `train_models.SvecdPINN` from `Codex - Frozen/train_models.py:172-226`.
5. The physics target is computed by `train_models.integrated_svecd()` at lines 229-236.
6. There is no single top-level “loss function.” The executed objective is assembled inside `train_variant()` at lines 306-368 from the local `state_huber()` and `total_huber()` closures and the inline onset, transition, trajectory, physics, monotonicity, and alpha-prior terms.

The frozen runner and config explicitly set:

| Setting | Frozen value |
|---|---:|
| dataset tag | `_sdr_dense_state` |
| feature tag | `_sdr_dense_state_obs30` |
| anchoring | `1` |
| external hybrid | `0` |
| ensemble members | `1` |
| physics weight | `0.1` |
| transition weight | `2.0` |
| trajectory weight | `0.25` |
| trajectory mode | `all` |
| severity weights LOW/MOD/HIGH | `0.7, 1.2, 3.0` |
| total-cracking weight | `0.4` |
| onset weight | `0.15` |
| cracked-row increment | `0.5` |
| Huber delta | `1.0` |

The following active values are **not recorded in `frozen_config.json` or the frozen output manifest**; they enter through code defaults:

| Setting | Effective value | Source |
|---|---:|---|
| epochs | 650 | `train_severity_state_pinn.py:63` |
| learning rate | 0.002 | line 64 |
| trajectory cap | 20,000 | line 69 |
| monotonicity weight | 0.1 | line 70 |
| alpha-prior weight | 0.1 | lines 71 and `train_models.py:150` |
| AdamW weight decay | `1e-4` | line 266 |
| multiband anchor | false | default used at lines 618/652 |
| self-anchor baseline | false | line 60 |

The 41 listed features become **44 network inputs** at run time because anchoring adds `LAST_OBS_CPCT_L`, `AGE_AT_LAST_OBS`, and `DT_SINCE_LAST` (`train_models.py:104-130`). Multiband anchoring is off, so `LAST_OBS_CPCT_M` and `LAST_OBS_CPCT_H` are not model inputs.

For the selected temporal fit, the verified full-batch shapes are:

| Tensor | Shape |
|---|---:|
| standardized training input `xb` | `(2008, 44)` |
| observed disjoint states `ys` | `(2008, 3)` |
| observed cumulative states `yc` | `(2008, 3)` |
| onset target, reduced strain, interval load, latent damage | `(2008,)` each |
| raw cumulative neural output | `(2008, 3)` |
| one-step transition pairs | 1,682 pairs; state tensors `(1682, 3)` |
| all-pairs trajectory set | 8,006 pairs; state tensors `(8006, 3)` |

Training is full-batch: one optimizer update per epoch using all training rows and all selected pairs. There is no minibatch loader.

# A. Verified mathematical specification

## A1. Input preprocessing and network outputs

For each split, each raw feature column is median-imputed using the training rows and standardized using the training-row mean and standard deviation:

\[
z_{ij}=\frac{\operatorname{imp}_{j}(x_{ij})-\mu_{j,\mathrm{tr}}}{\sigma_{j,\mathrm{tr}}}.
\]

The shared trunk is

\[
h_i = \operatorname{SiLU}\!\circ\operatorname{LN}\!\circ\operatorname{FC}_{64}
\circ\operatorname{SiLU}\!\circ\operatorname{LN}\!\circ\operatorname{FC}_{64}
\circ\operatorname{SiLU}\!\circ\operatorname{LN}\!\circ\operatorname{FC}_{32}(z_i),
\]

where the actual implementation is three blocks `FC -> LayerNorm -> SiLU` with widths `64,64,32`.

The two direct neural heads are

\[
S_i=\operatorname{softplus}(w_S^T h_i+b_S)\ge 0,
\qquad
o_i=w_o^T h_i+b_o.
\]

`S` is a transformed direct network output, not a recursively accumulated neural state. `o` is an onset logit. The onset logit does **not** gate or multiply the cracking prediction.

## A2. Ordered cumulative severity transfer

The trainable transfer parameters are

\[
C_1=e^{\ell_{C_1}}>0,\quad
C_2=\operatorname{softplus}(r_{C_2})>0,\quad
\beta_1=\operatorname{softplus}(r_{\beta_1})>0,
\quad A_{\max}=\operatorname{softplus}(r_A)>0.
\]

The thresholds are

\[
\beta_{0,L}=b_0,\quad
\beta_{0,M}=b_0+\operatorname{softplus}(d_1),\quad
\beta_{0,H}=b_0+\operatorname{softplus}(d_1)+\operatorname{softplus}(d_2).
\]

Thus \(\beta_{0,L}\le\beta_{0,M}\le\beta_{0,H}\). Define

\[
f_i=\operatorname{clamp}\left(C_1(S_i+10^{-9})^{C_2},10^{-6},1\right),
\qquad g_i=\log_{10}(f_i).
\]

The three raw model cracking outputs are cumulative severity-or-greater states:

\[
E_{i,b}=A_{\max}\,\sigma\{\beta_1(g_i-\beta_{0,b})\},
\qquad b\in\{L,M,H\}.
\]

Positive \(\beta_1\), ordered thresholds, and the common \(A_{\max}\) guarantee

\[
E_{i,L}\ge E_{i,M}\ge E_{i,H}>0
\]

before the defensive reconstruction. Ordering is therefore guaranteed by architecture/parameterization, then re-enforced algebraically.

For training losses, `cum_to_state_torch()` first clamps every cumulative component to `[0,100]` and applies a running minimum:

\[
\bar E_L=\operatorname{clamp}(E_L,0,100),\quad
\bar E_M=\min\{\bar E_L,\operatorname{clamp}(E_M,0,100)\},
\]
\[
\bar E_H=\min\{\bar E_M,\operatorname{clamp}(E_H,0,100)\}.
\]

The disjoint predictions are

\[
\hat y_L=\max(\bar E_L-\bar E_M,0),\quad
\hat y_M=\max(\bar E_M-\bar E_H,0),\quad
\hat y_H=\max(\bar E_H,0).
\]

The cumulative observed targets are

\[
y^c_L=y_L+y_M+y_H,\qquad y^c_M=y_M+y_H,\qquad y^c_H=y_H,
\]

clipped to `[0,100]` when the target dataset is constructed. Total cracking is not predicted by a separate head: it is the cumulative LOW-or-higher output \(E_L\), and the reconstructed disjoint states telescope to \(\bar E_L\).

## A3. Huber kernel, band weights, and row weights

With frozen \(\delta=1\), PyTorch's Huber kernel is

\[
H_1(e)=
\begin{cases}
\tfrac12e^2,&|e|<1,\\
|e|-\tfrac12,&|e|\ge 1.
\end{cases}
\]

All supervised magnitude errors are evaluated in `log1p` space. Let

\[
a=(a_L,a_M,a_H)=(0.7,1.2,3.0)
\]

and let the row weight be

\[
q_i=1+0.5\,\mathbf 1(y_{i,L}+y_{i,M}+y_{i,H}>0).
\]

Therefore uncracked rows have weight 1 and cracked rows have weight **exactly 1.5**. The code implements this by adding `0.5` to the base weight, but it is not a separate auxiliary cracked-only loss.

For a row set \(R\), the severity-state loss is

\[
\mathcal L_{\mathrm{state}}(R)=
\frac{1}{3|R|}\sum_{i\in R}\sum_{b\in\{L,M,H\}}
q_i a_b H_1\!\left[\log(1+\hat y_{i,b})-\log(1+y_{i,b})\right].
\]

The division is by all `3|R|` entries after multiplication; the band and row weights are not normalized to sum to one.

The total-cracking loss is

\[
\mathcal L_{\mathrm{total}}(R)=
\frac1{|R|}\sum_{i\in R}q_i
H_1\!\left[\log(1+E_{i,L})-\log(1+y^c_{i,L})\right].
\]

Unlike the state conversion, the base total loss reads raw `cum[:,0]` directly; it does not explicitly clamp it to 100.

## A4. Onset loss

The target and neural prediction are

\[
t_i=\mathbf 1(y_{i,L}+y_{i,M}+y_{i,H}>0),\qquad p_i=\sigma(o_i).
\]

The active onset loss is mean binary cross-entropy with logits:

\[
\mathcal L_{\mathrm{onset}}=-\frac1n\sum_i
\left[t_i\log\sigma(o_i)+(1-t_i)\log(1-\sigma(o_i))\right].
\]

The onset head is auxiliary to training and does not alter the severity outputs. In the frozen trainer's reported metrics, `OnsetAUC` is computed from predicted total cracking, not from \(p_i\). The later `Codex - FinalValidation` utilities instead use \(p_i\). These are different onset scores.

## A5. One-step transition anchoring

Rows are grouped by `SECTION_ID`, sorted by `AGE`, and each row after the first is paired with the immediately preceding survey. A transition pair is retained only when both rows are in the training subset.

For current row \(i\) and previous row \(p(i)\), let \(E_i\) and \(E_{p(i)}\) be the three cumulative network outputs. The frozen training path has `SELF_ANCHOR_BASELINE=false`, so `prev_base = cum[prev_pos]`. The anchored cumulative prediction is

\[
\tilde E^{\mathrm{tr}}_i
=y^c_{p(i)}+\operatorname{ReLU}(E_i-E_{p(i)}),
\]

where ReLU is applied **element by element to all three cumulative severity components**, not to disjoint states, total cracking alone, the onset logit, or latent \(S\).

`cum_to_state_torch()` then clamps/reorders \(\tilde E^{\mathrm{tr}}_i\) and converts it to disjoint LOW/MOD/HIGH states. The transition loss is

\[
\mathcal L_{\mathrm{trans}}
=\mathcal L_{\mathrm{state}}\left(Q(\tilde E^{\mathrm{tr}}),y_{\mathrm{current}}\right),
\]

where \(Q\) denotes the defensive cumulative-to-disjoint conversion. There is no transition-specific total-cracking term.

If a predicted cumulative component decreases, its ReLU increment is zero and that component remains at its previous observed cumulative value. Different components can receive different increments; if this breaks cumulative ordering, the running-minimum reconstruction restores ordering before the disjoint state loss. Although each anchored cumulative component cannot decrease below its own observed anchor, disjoint LOW or MODERATE area may decrease as area moves to a higher severity.

## A6. All-pairs trajectory loss

Within every section, training rows are sorted by `AGE`. In `all` mode, every index pair \((a,f)\) with the anchor position earlier than the future position is constructed. The general code deterministically subsamples to at most 20,000 pairs with RNG seed 0, but the cap is inactive: there are 8,006 temporal-training pairs and 9,944 pairs in the complete model-ready data.

For every pair, both the anchor and future feature rows are rebuilt to use the **same observed anchor**. With the frozen single-band anchor feature set this means:

- `LAST_OBS_CPCT_L` is set to observed \(y^c_{a,L}\);
- `AGE_AT_LAST_OBS` is set to anchor age;
- `DT_SINCE_LAST` is zero for the anchor and future-age minus anchor-age for the future.

The M/H anchor columns are absent from the actual feature matrix.

Let \(E_f^{(a)}\) be the network output for the future feature row anchored at \(a\), and \(E_a^{(a)}\) the output for the self-anchored anchor row. Then

\[
\tilde E^{\mathrm{traj}}_{a,f}
=y^c_a+\operatorname{ReLU}\left(E_f^{(a)}-E_a^{(a)}\right).
\]

The active trajectory term is

\[
\mathcal L_{\mathrm{traj}}
=\mathcal L_{\mathrm{state}}\left(Q(\tilde E^{\mathrm{traj}}),y_f\right)
+0.4\,\mathcal L_{\mathrm{total}}\left(\tilde E^{\mathrm{traj}},y^c_f\right).
\]

The total subterm reads raw first cumulative component before `Q`, while the state subterm uses the clamped/reordered disjoint conversion.

Every generated pair has equal base averaging weight. Cracked-future and band weights still apply. Sections with more surveys contribute quadratically more pairs, future rows appear once for every earlier anchor, and there is no per-section or per-future-row normalization. This differs from the transition loss, which uses only the single immediately preceding survey for each eligible current row.

## A7. Actual reduced-strain calculation

The trainer does not calculate reduced strain from feature names. It reads the precomputed `EPS_R` column (`train_models.py:144`). Its upstream executed dataset formula is in `build_dataset.py:227-290`.

For section-specific reference modulus

\[
E_R=|E^*(20^\circ\mathrm C,10\,\mathrm{Hz})|,
\]

monthly pavement temperature \(T_m\), load bin \(l\), normalized load-spectrum weight \(w_l\), LEA tensile microstrain \(\varepsilon^{\mu\varepsilon}_{t,m,l}\), and \(\beta=3.949\), the bin-level reduced strain is

\[
\varepsilon_{R,m,l}
=\frac{|E^*(T_m,10\,\mathrm{Hz})|}{E_R}
\varepsilon^{\mu\varepsilon}_{t,m,l}\,10^{-6}.
\]

The single section-level `EPS_R` used at every survey is the power mean

\[
\mathrm{EPS\_R}
=\left[
\frac{\sum_m\sum_l w_l\varepsilon_{R,m,l}^{3.949}}
{\sum_m\sum_l w_l}
\right]^{1/3.949}.
\]

This seasonal/load-spectrum power-mean formula is more precise than writing only \(\varepsilon_R=(E^*/E_R)\varepsilon_t\). The latter is true per bin but is not the final scalar fed to training.

## A8. Executed accumulated physics target

First, the loaded cumulative axle count is converted into nonnegative interval load:

\[
\Delta N_{s,k}
=\frac{1}{10^6}\max\left(N_{s,k}-N_{s,k-1},0\right),
\]

with the first row in a section using \(N_{s,1}/10^6\) rather than a difference. This is computed before split selection in `train_models.py:93-95,145`.

In the operational frozen model, write \(\kappa=\texttt{log\_kscale}\). Then

\[
\alpha=1+\operatorname{softplus}(r_\alpha)>1,
\qquad \kappa\in\mathbb R.
\]

The interval increment actually used by `integrated_svecd()` is

\[
\Delta W_{s,k}
=\Delta N_{s,k}\,
\left[\max\left(\tfrac12\varepsilon_{R,s,k}^2,10^{-30}\right)\right]^{\alpha/(1+\alpha)}
e^\kappa.
\]

The accumulated target is

\[
W_{s,k}=\sum_{j\le k}\Delta W_{s,j},
\]

where rows passed to the training function are lexicographically sorted by section and age, cumulatively summed, offset at section boundaries, and restored to original training-row order.

This computation resets the accumulator only at the first included row of each `SECTION_ID`. `SECTION_ID` is `STATE_CODE_SHRP_ID`, not construction number. It does **not** reset at maintenance, rehabilitation, `CONSTRUCTION_NO` changes, load resets, or observed cracking drops. A negative cumulative-load difference is merely clipped to zero, after which prior accumulated damage remains. The direct neural \(S_i\) also has no reset operation.

Most importantly, the executed increment is **not** the standalone `svecd.py` formula

\[
\left[\tfrac12\varepsilon_R^2(-dC/dS)\right]^{\alpha/(1+\alpha)}.
\]

`integrated_svecd()` omits \(-dC/dS=C_1C_2S^{C_2-1}\), has no dependence on prior damage state, and performs a simple cumulative sum of a strain/load power. Therefore the frozen constraint is accurately described as an **S-VECD-informed accumulated damage proxy**, not as exact numerical integration of the S-VECD evolution equation documented in `svecd.py:14-19,50-59,88-109`.

## A9. Physics loss and parameter effects

The active physics loss is

\[
\ell^S_i=\log(S_i+10^{-6}),\qquad
\ell^W_i=\log(W_i+10^{-9}),
\]

\[
\mathcal L_{\mathrm{phys}}=
\frac1n\sum_{i=1}^n
\left[(\ell^S_i-\bar\ell^S)-(\ell^W_i-\bar\ell^W)\right]^2.
\]

Centering is global over the complete full-batch training-row tensor. It is not by section, sequence, trajectory, or minibatch.

Parameter audit:

| Parameter | Trainable? | Appears in executed physics target? | Nonzero mathematical effect? |
|---|---|---|---|
| neural weights producing `S` | yes | physics loss left side | yes |
| `raw_alpha` / `alpha` | yes | exponent `alpha/(1+alpha)` | yes; also alpha-prior loss |
| `log_kscale` | yes | global factor `exp(kappa)` | only through the `+1e-9` inside `log(W+1e-9)` and AdamW; otherwise canceled exactly by mean-centering |
| `C1` | yes | no | yes in cracking transfer, not in accumulated physics |
| `C2` | yes | no | yes in cracking transfer, not in accumulated physics |
| `b0`, threshold gaps, `beta1`, `Amax` | yes | no | yes in cracking transfer |
| reduced strain, interval load | fixed tensors | yes | yes |
| factor `0.5`, clamps and epsilons | fixed | yes | yes |

To see the `k_scale` cancellation, without the additive epsilon

\[
\log(e^\kappa W_i^0)-\operatorname{mean}_j\log(e^\kappa W_j^0)
=\log W_i^0-\operatorname{mean}_j\log W_j^0.
\]

Thus a global multiplicative scale cancels exactly. Because the code uses \(\log(e^\kappa W_i^0+10^{-9})\), `log_kscale` is not symbolically dead, but its effect is negligible whenever \(e^\kappa W_i^0\gg10^{-9}\). It is mathematically near-redundant and should not be interpreted as an identified damage scale.

There are further practical identifiability redundancies in the transfer. Away from clamps,

\[
g=\log_{10}C_1+C_2\log_{10}(S+10^{-9}),
\]

so shifts in \(\log_{10}C_1\) can be absorbed by the trainable thresholds, and slope changes can be traded between \(C_2\) and \(\beta_1\). Each parameter has a local mathematical effect, but the individual values are not uniquely mechanistically identifiable from this transfer alone.

## A10. Monotonicity loss

At every epoch, the code computes

\[
G=\frac{\partial\sum_iS_i}{\partial z},
\]

which, because rows are processed independently by the network, gives rowwise derivatives \(G_{i,j}=\partial S_i/\partial z_{i,j}\). The active penalty is

\[
\mathcal L_{\mathrm{mono}}
=\frac1n\sum_i\operatorname{ReLU}\left(-\frac{\partial S_i}{\partial z_{i,\mathrm{AGE}}}\right)
+\frac1n\sum_i\operatorname{ReLU}\left(-\frac{\partial S_i}{\partial z_{i,\mathrm{N\_CYCLES}}}\right).
\]

Both AGE and cumulative `N_CYCLES` are active. Differentiation is with respect to **imputed, standardized input coordinates**, not original-scale AGE or load. A positive standard deviation preserves derivative sign, so the sign constraint corresponds to the original variable sign, but the penalty magnitude is scale-dependent. The differentiated quantity is latent neural damage \(S\), not cumulative cracking, disjoint severity, onset probability, or the integrated target \(W\). It is evaluated during every one of the 650 full-batch epochs.

## A11. Alpha-prior loss

The training-subset prior is

\[
\alpha_0=\operatorname{nanmedian}_{i\in\mathrm{train}}(\mathrm{ALPHA\_PRIOR}_i).
\]

Upstream, `ALPHA_PRIOR` was constructed as

\[
\mathrm{ALPHA\_PRIOR}_i
=\operatorname{clip}\left(1+\frac1{m_{\max,i}},2.0,3.5\right),
\]

where \(m_{\max}=-\texttt{master\_curve.alpha}\times\texttt{gamma}/4\). For the frozen temporal training rows, the verified median is exactly \(\alpha_0=3.5\). The active loss is

\[
\mathcal L_\alpha=(\alpha-\alpha_0)^2.
\]

The frozen parameterization does not cap \(\alpha\) at 3.5; it only enforces \(\alpha>1\). The squared prior is a soft pull, not a bound.

## A12. Complete frozen objective

For the selected configuration, the exact scalar assembled before backpropagation is

\[
\boxed{
\begin{aligned}
\mathcal L={}&
\mathcal L_{\mathrm{state}}
+0.4\mathcal L_{\mathrm{total}}
+0.15\mathcal L_{\mathrm{onset}}\\
&+2.0\mathcal L_{\mathrm{trans}}
+0.25\left(\mathcal L_{\mathrm{traj,state}}+0.4\mathcal L_{\mathrm{traj,total}}\right)\\
&+0.1\mathcal L_{\mathrm{phys}}
+0.1\mathcal L_{\mathrm{mono}}
+0.1\mathcal L_{\alpha}.
\end{aligned}}
\]

Equivalently, the effective coefficient of the trajectory total subterm is `0.25 x 0.4 = 0.1`.

AdamW additionally applies decoupled weight decay `1e-4` to all parameters in `pinn.parameters()`, including neural and physics/transfer parameters. Decoupled AdamW weight decay is an optimizer update, not an explicitly added scalar regularization term in the reported \(\mathcal L\). A cosine-annealing learning-rate scheduler runs for 650 epochs. There is no dropout, early stopping, minibatching, explicit L1 term, standalone trajectory-pair weighting, or active external residual learner.

# B. Implementation mapping table

| Equation or operation | Code file | Function/class | Relevant lines | Configuration value | Notes |
|---|---|---|---:|---|---|
| Frozen process entry | `04 - Agent Codes/Codex - Frozen/run_frozen_best.py` | `main` | 28-57 | runner environment | Launches the actual trainer |
| Frozen machine-readable config | `04 - Agent Codes/Codex - Frozen/frozen_config.json` | n/a | 1-55 | listed weights | Omits several active defaults |
| Dataset load/filter/sort | `04 - Agent Codes/Codex - Frozen/train_models.py` | module initialization | 50-101 | dataset/feature tags | 2,285 model-ready rows, 326 sections |
| Runtime anchor features | same | module initialization, `build_X` | 104-143 | anchor=1; multiband=false | 41 listed + 3 runtime inputs = 44 |
| Interval axle count | same | module initialization | 93-95, 145 | divide by `1e6` | Negative differences clipped to zero |
| Split-wise imputation/scaling | `train_severity_state_pinn.py` | `scaled_matrices` | 164-178 | median + StandardScaler | Fit only on training rows |
| Prior-survey index | same | `previous_index` | 104-112 | n/a | Group by section, order by AGE |
| All trajectory pairs | same | `trajectory_pairs` | 195-228 | mode all, cap 20,000 | Actual cap inactive |
| Same-anchor trajectory matrices | same | `anchored_rows_matrix`, `make_trajectory_data` | 181-192, 231-246 | single L anchor | M/H anchor columns absent |
| Model architecture | `train_models.py` | `SvecdPINN.__init__` | 172-198 | fixed 64,64,32 | LayerNorm + SiLU |
| Alpha parameterization | same | `SvecdPINN.alpha` | 199-202 | inherited default | `1 + softplus(raw_alpha)`, no upper bound |
| Ordered thresholds | same | `SvecdPINN.thresholds` | 203-208 | trainable | Softplus nonnegative gaps |
| Damage-to-cumulative transfer | same | `SvecdPINN.transfer` | 210-219 | trainable | Clamp, log10, sigmoid, shared Amax |
| Latent damage and onset heads | same | `SvecdPINN.forward` | 221-226 | n/a | Softplus S; independent onset logit |
| Section cumulative physics proxy | same | `segment_index`, `integrated_svecd` | 158-166, 229-236 | `w_phys=0.1` | Cumulative sum resets at section boundary only |
| Reduced-strain upstream formula | `04 - Agent Codes/Codex - Second/build_dataset.py` (same provenance family copied into frozen artifact) | `main` | 227-290 | beta 3.949 | Seasonal/load-spectrum power mean |
| Alpha-prior upstream formula | `04 - Agent Codes/Codex - Second/augment_features.py` | `relax_table` | 31-38 | clip 2.0,3.5 | Dataset metadata, not a model input |
| Huber/BCE objects | `train_severity_state_pinn.py` | `train_variant` | 265-269 | delta 1.0 | Both reduction conventions audited above |
| Cracked-row weight | same | `train_variant` | 294-295 | increment 0.5 | Effective cracked weight 1.5 |
| State Huber | same | local `state_huber` | 306-310 | 0.7,1.2,3.0 | Mean over rows and bands |
| Total Huber | same | local `total_huber` | 312-316 | total coefficient 0.4 | Uses raw cumulative L output |
| Base supervised + onset objective | same | `train_variant` | 319-325 | 1, 0.4, 0.15 | Full batch |
| Transition anchor/loss | same | `train_variant` | 327-337 | coefficient 2.0 | Elementwise ReLU on cumulative components |
| Trajectory anchor/loss | same | `train_variant` | 339-347 | coefficient 0.25 | Includes nested total coefficient 0.4 |
| Physics log-centering loss | same | `train_variant` | 349-355 | coefficient 0.1 | Global training-row centering |
| Monotonicity loss | same | `train_variant` | 357-361 | coefficient 0.1 | S wrt standardized AGE and N_CYCLES |
| Alpha-prior loss | same | `train_variant` | 363-366 | coefficient 0.1 | Temporal prior median 3.5 |
| Optimizer regularization | same | `train_variant` | 265-267, 368-370 | AdamW wd `1e-4` | Decoupled, not in scalar loss |
| Cumulative-to-disjoint conversion | same | `cum_to_state_torch` | 138-148 | clamp 0,100 | Running-minimum ordering safeguard |
| Temporal transition inference | same | `predict_transition` | 391-402 | selected reporting mode | Same observed-anchor + ReLU construction |
| Frozen metric onset score | same | `metrics` | 405-447 | n/a | AUC uses predicted total, not onset head |
| Frozen manifest writer | same | `write_report`/`write_temporal_report` | 615-674 | partial | Does not record mono/alpha/epochs/lr/wd |
| Later 128 candidate runner | `04 - Agent Codes/Codex - Second/run_trajectory_sampling_sweep.py` | `run_one` | 42-89 | 128,128,64; alpha 2-3.5; cap 0 | Development candidate, not frozen runner |
| Later bounded-alpha implementation | `04 - Agent Codes/Codex - Second/train_models.py` | `_bounded`, `SvecdPINN.alpha` | 155-180, 225-307 | alpha 2.0,3.5 | Used only when env bounds are supplied |

# C. Manuscript-safe formulation

Only the following equations should appear in the main manuscript for the operational frozen paper model:

1. Disjoint-to-cumulative target definitions:
   \[
   y^c_L=y_L+y_M+y_H,\quad y^c_M=y_M+y_H,\quad y^c_H=y_H.
   \]
2. Latent damage and ordered cumulative transfer:
   \[
   S=\operatorname{softplus}(f_\theta(z)),\quad
   E_b=A_{\max}\sigma\{\beta_1[\log_{10}(\operatorname{clamp}(C_1(S+10^{-9})^{C_2},10^{-6},1))-\beta_{0,b}]\},
   \]
   with positive \(C_1,C_2,\beta_1,A_{\max}\) and ordered \(\beta_{0,L}\le\beta_{0,M}\le\beta_{0,H}\).
3. Cumulative-to-disjoint reconstruction:
   \[
   \hat y_L=E_L-E_M,\quad \hat y_M=E_M-E_H,\quad \hat y_H=E_H,
   \]
   accompanied by text that implementation clamps to `[0,100]` and re-enforces ordering first.
4. Executed accumulated damage proxy:
   \[
   W_{s,k}=\sum_{j\le k}\frac{\max(N_{s,j}-N_{s,j-1},0)}{10^6}
   \left[\max\left(\tfrac12\varepsilon_{R,s,j}^2,10^{-30}\right)\right]^{\alpha/(1+\alpha)}e^\kappa.
   \]
5. Globally centered log physics loss:
   \[
   \mathcal L_{\mathrm{phys}}=\operatorname{mean}\{[(\log(S+10^{-6})-\overline{\log(S+10^{-6})})-(\log(W+10^{-9})-\overline{\log(W+10^{-9})})]^2\}.
   \]
6. Elementwise cumulative transition anchor:
   \[
   \tilde E_i=y^c_{p(i)}+\operatorname{ReLU}(E_i-E_{p(i)}).
   \]
7. All-pairs trajectory anchor:
   \[
   \tilde E_{a,f}=y^c_a+\operatorname{ReLU}(E_f^{(a)}-E_a^{(a)}),\quad a<f.
   \]
8. Log1p Huber state loss with row/band weights, onset BCE, monotonicity term, and the complete weighted objective from A12.

In the main text, call \(W\) an “S-VECD-informed accumulated damage proxy.” Do not claim that the implemented constraint numerically integrates the full characteristic-curve S-VECD law.

# D. Supplementary formulation

The supplement should report:

1. The seasonal/load-spectrum power-mean reduced-strain equation with `beta=3.949`, reference temperature 20 C, reference frequency 10 Hz, and LEA microstrain conversion by `1e-6`.
2. Median imputation and training-only standardization, including the fact that monotonicity derivatives are taken with respect to standardized coordinates.
3. The exact Huber kernel (`delta=1`), `log1p` transform, unnormalized band weights, and row weight `1 + 0.5 I(cracked)`.
4. Defensive cumulative clamping/running-minimum logic and the fact that raw `cum[:,0]` is used in total losses.
5. Pair-generation rules, counts (1,682 transitions and 8,006 temporal trajectory pairs), equal-pair averaging, quadratic section contribution, deterministic cap behavior, and absence of section-balanced weighting.
6. The three runtime anchor features and absence of M/H anchor features.
7. Full physics epsilons and clamps: `1e-30` before the physics power, `1e-9` in `log(W+...)`, `1e-6` in `log(S+...)`, and transfer clamps `[1e-6,1]`.
8. Section/age sorting, section-boundary resets, lack of construction/maintenance resets, and clipping negative load increments to zero.
9. Alpha-prior construction, training-subset median, active coefficient 0.1, and the operational frozen parameterization `alpha > 1` without an upper bound.
10. AdamW (`lr=0.002`, weight decay `1e-4`), cosine annealing, 650 full-batch epochs, seed 0, CPU execution, and lack of a serialized checkpoint.
11. The onset-head distinction: BCE trains the neural onset probability, frozen headline AUC uses total cracking as its score, and July final validation uses the neural onset head.
12. Parameter identifiability: global `k_scale` cancellation under log mean-centering and confounding among `C1`, thresholds, `C2`, and `beta1`.

# E. Corrections required

## E1. Final-model identity must be resolved

- `Codex - Frozen`, `Codex - Frozen GitHub`, the Results draft, and `Codex - FinalValidation` define the paper model as the older 64-64-32 configuration with temporal `R2log_total=0.788747`.
- `Codex - Second/EXPERIMENT_MEMORY.md:5-31` defines a later “Current Best Kept Setup” as 128-128-64 with bounded alpha and `R2log_total=0.798006`.
- `PAPER_PACKAGE/03_Methods_Outline.md:69-74` describes the later 128-wide/bounded-alpha model, while `PAPER_PACKAGE/04_Results_Draft.md:27-47` reports and ablates the older frozen model.
- These cannot be combined in one manuscript as if they were one experiment. Either retain the operational frozen model and correct Methods, or formally refreeze and rerun all final validation, baselines, limited-data experiments, ablations, and manuscript tables with the later candidate.

## E2. Architecture and alpha constraints

- For the currently reported/final-validated results, `hidden layers 128,128,64` is inaccurate; executed frozen layers are `64,64,32`.
- For the currently reported/final-validated results, `SVECD_ALPHA_MIN=2.0` and `SVECD_ALPHA_MAX=3.5` are inaccurate. The executed constraint is `alpha = 1 + softplus(raw_alpha)`, plus a soft squared prior toward the training median (3.5 in the temporal fit).
- The later development candidate does use the sigmoid bound `alpha = 2.0 + 1.5 sigmoid(raw_alpha)`, but those are not the frozen reported results.

## E3. Governing physics equation

- Any manuscript equation containing `(-dC/dS)` in the **executed training accumulation** is inaccurate. The final training path omits it.
- Any claim that `C1` or `C2` affects accumulated physics damage is inaccurate. They affect only the neural damage-to-cracking transfer.
- Any recursive equation `S_k = S_{k-1} + Delta S(S_{k-1})` is inaccurate for the training constraint. The code accumulates a state-independent proxy `W` and separately predicts direct rowwise neural `S`.
- Calling the executed equation exact S-VECD integration is unsupported. “S-VECD-informed accumulated damage proxy” is safe.

## E4. Scaling and identifiability

- A statement that `k_scale` “sets the absolute damage level” is misleading under the actual centered-log physics loss. Its global multiplicative effect cancels exactly without the `1e-9` epsilon and is only epsilon-sensitive in code.
- Individual `C1`, `C2`, threshold, and `beta1` values should not be given a uniquely identified mechanistic interpretation because the transfer contains shift/slope confounding.

## E5. Damage resetting and grouping

- A statement that damage resets after rehabilitation, maintenance, construction-number change, cumulative-load reset, or cracking reset is false. It resets only at `SECTION_ID` boundaries in the selected row tensor.
- `SECTION_ID` excludes `CONSTRUCTION_NO`; many physical sections contain more than one construction number in the broader dataset.
- A simplified accumulated equation should not omit the clipping of negative load increments or the division by one million.

## E6. Loss descriptions

- “Cracked rows receive weight 1.5” is exactly correct. The implementation computes this as base 1 plus 0.5 times the cracked indicator; it is not a separate cracked-only auxiliary loss.
- Transition loss contains only severity-state Huber. It does not contain the total-cracking Huber subterm.
- Trajectory loss contains both severity-state Huber and `0.4` times total Huber, all multiplied by `w_traj=0.25`.
- Total cracking is not a separate neural output. It is cumulative LOW-or-higher cracking `cum[:,0]`.
- The total-loss input is not always the clamped/reordered cumulative tensor; it reads the raw first cumulative component.
- Pair losses are not section balanced. “All pairs are averaged equally” is correct only before cracked-row and severity-band weighting.

## E7. Monotonicity

- The differentiated output is latent neural `S`, not predicted cracking.
- Gradients are with respect to normalized AGE and cumulative `N_CYCLES`, not original units.
- Both variables are active in every full-batch epoch.
- The monotonicity penalty does not itself guarantee that disjoint LOW or MODERATE severity areas increase; the ordered cumulative representation allows them to decrease during severity migration.

## E8. Onset reporting

- The frozen headline `OnsetAUC=0.866` in `RESULTS_SEVERITY_STATE_PINN.md` is based on predicted total cracking as the ranking score (`metrics`, lines 409-427), not on the independently trained onset probability.
- `Codex - FinalValidation` uses the neural onset-head probability. Manuscript tables must label which score is used and must not mix these AUCs.
- The onset head is not a hurdle gate for severity. Describing the frozen severity prediction as `P(onset) x severity` is false.

## E9. Reproducibility documentation

- `Codex - Frozen GitHub/REPRODUCIBILITY.md:20` calls all 2,879 dataset rows “model-ready.” The trainer filters to 2,285 model-ready rows across 326 sections; 2,879 is the stored processed dataset row count.
- Frozen config/manifest files omit active monotonicity weight, alpha-prior weight, architecture, epochs, learning rate, weight decay, trajectory cap, and alpha parameterization. A public config must record them or explicitly declare inherited defaults.
- The artifact contains no checkpoint, so the phrase “frozen model” means a frozen recipe/dataset, not frozen learned weights. Numerical reproduction is a retraining experiment.

## E10. Dead or inactive code and parameters

- `svecd.py` is not imported by the frozen trainer. Its deterministic `dS_dN()` and `accumulate_damage()` equations are reference/dead code for the final execution path.
- `train_models.train_pinn()` is not the training function used by the severity-state frozen entry point.
- Hurdle and external hybrid branches are inactive (`SVECD_HYBRID=0`; the severity trainer does not gate by onset).
- Ensemble setting is 1, so there is no ensemble averaging in the selected run.
- `SVECD_MULTIBAND_ANCHOR` and `STATE_PINN_SELF_ANCHOR_BASELINE` are false.
- The later `Codex - Second` generic physics-parameter-prior loss exists but is inactive in the later kept candidate (`w_param_prior=0`) and does not exist in the operational frozen trainer.
- The trajectory cap is configured but inactive because every relevant all-pairs set is below 20,000.
- `log_kscale` is trainable but nearly mathematically redundant because of global log mean-centering.

## Final publication decision

The current manuscript formulation is **not fully accurate**. The safest immediate paper path is to retain the operational frozen model because it is the one used for the complete downstream evidence, then correct the Methods to `64,64,32`, `alpha>1` with a 0.1 soft alpha prior, and the exact S-VECD-informed proxy equations above. Choosing the later 128-wide/bounded-alpha candidate instead requires a new formal freeze and regeneration of every result that currently imports or compares against `Codex - Frozen`.
