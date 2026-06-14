# SCF iteration-count reduction — strategy catalogue (resume point)

**Status**: analysis / planning only, none implemented (2026-06-14). This is the
master catalogue for lowering the *number* of SCF iterations, written after the
per-iteration BLAS/LAPACK path was proven near-floor (see
[`../../home`-memory `initial_guess_blas_2026_06_14`] and the note at the bottom).
Per-iter wall is at its limit; the only remaining large lever is **doing fewer
iterations**. Four of these strategies already have deep write-ups in
[`heme_scf_advanced_methods_2026_05_28.md`](heme_scf_advanced_methods_2026_05_28.md)
— this doc is the complete landscape and the agreed sequencing.

---

## 0. Why iteration count is the lever now

- Per-iteration cost is at the BLAS/LAPACK floor: `dsyevd` beats every subset
  eigensolver at M=364; `DGEMM` beats `DSYMM`/`DGEMMT`; the DIIS algebra is
  minimal and ulp-locked to heme. (Measured 2026-06-14.)
- LIO is **primarily a QM/MM MD code** → every saved iteration multiplies over
  thousands of MD steps. A method that only helps single-points is worth far
  less than one that helps every BOMD step.
- Two hard constraints gate *every* trajectory-changing idea:
  1. **heme iter-count is Lyapunov-chaotic.** Any change must be judged on the
     **median across `OPENBLAS_NUM_THREADS ∈ {1,4,6,8}`**, never the lucky
     default. A "win" on default-72 that is 200 at OMP=8 is not a win.
  2. **float32 GPU-XC noise** is a real convergence floor (~5–7 iters of
     criterion-noise tail). Some methods fight it; some are limited by it.
- **No-toggle rule (manuel):** a method ships unconditionally or stays out.
  Nothing that helps fosfato but regresses heme is acceptable.

## Reference baselines (current binary, `LIO_OVERLAP_INT3LU_G2G=1`)

| Case | Spin | M / NCO | Guess | conver | Iters | Notes |
|------|------|---------|-------|--------|-------|-------|
| fosfato (03) | closed | 364 / 99 | 1e | DIIS(2) | **25** | well-behaved, ~23 ms/iter |
| heme (13) | open NUNP=4 | basis-locked / — | VCINP restart | DIIS(2) | **72 (lucky) / 72–186 spread** | chaotic, ~270 ms/iter, E=−2829.6326±1µHa |

What LIO already has (so we don't rebuild it):
- Accelerators: damping(1), **DIIS(2, default)**, hybrid(3), biased-DIIS(4/5),
  **EDIIS(6)**; staged thresholds `EDIIS_start=1e-20`, `DIIS_start=1e-2`,
  `bDIIS_start=1e-3`; `nDIIS=15`; `tolD=1e-6`; `DIIS_bias=1.05`.
- **Level shifting**: present but `level_shift=.false.`, gated at
  `HL_gap < lvl_shift_cut=0.005` (heme's real d-d gap ≈ 0.0615 → **never fires**),
  `lvl_shift_en=0.25`.
- Guesses: 1e core-Hamiltonian (default) + crude aufbau diagonal
  (`initial_guess=1`). **No SAD/SAP.**
- MD reuse: `VCINP` reuses *last-step* density only. **No cross-step
  extrapolation.**
- **No** fractional occupation/smearing, **no** orbital following, **no**
  second-order SCF.
- `do_rho_ls` linear-search infrastructure exists (cousin of ODA).
- `converger_subs/` ≈ 1700 LOC; `initial_guess.f90` ≈ 290 LOC.

---

# Family A — Better starting point (single-point + MD step 1)

## A1. SAD — Superposition of Atomic Densities
**Mechanism.** Build the guess density block-diagonally from precomputed,
spherically-averaged neutral-atom densities (one tiny atomic SCF per element,
cached, or shipped as tables). P⁰ = ⊕_atoms P_atom.
**Why for LIO.** The default 1e guess ignores electron–electron repulsion
entirely → poor. SAD is the modern default in ORCA/Psi4/Q-Chem. fosfato-class
main-group: expect **25 → ~15–18 iters**.
**Risk.** Low — it is a *guess*; the converged result is identical. Only iter
count moves. Weak for transition metals (atomic ground-state multiplet
ambiguity) → see SAD-d below for heme.
**float32/heme.** Neutral; for heme, plain SAD's Fe atomic occupation ambiguity
limits it — the d-shell-aware **SAD-d** variant is the relevant one and is
written up in detail in `heme_scf_advanced_methods_2026_05_28.md §4`
(conclusion there: SAD-d won't run under heme's `VCINP=t`, and an aufbau d-shell
guess was 391 iters vs restart 72 — so SAD helps *fresh* runs, not heme-restart).
**Effort.** Medium — atomic density source + assembly into `initial_guess.f90`.
**Lands in.** `lioamber/initial_guess.f90` (new `initial_guess_sad`), tables or
per-element atomic SCF helper.
**Applies to.** Fresh single-points + MD step 1 only (restart steps skip it).

## A2. SAP — Superposition of Atomic Potentials (Lehtola 2019)
**Mechanism.** Assemble a guess Fock from tabulated atomic *effective radial
potentials* (GRASP-derived, published), add to the kinetic+nuclear core, then
one diagonalization. No atomic SCF, no occupation ambiguity.
**Why for LIO.** More robust than SAD and **better for metals** → the more
promising guess for heme's Fe than plain SAD. Single extra one-electron-style
integral over the SAP potential.
**Risk.** Low (guess only). Needs the SAP potential tables shipped.
**Effort.** Medium — radial-potential evaluation on the grid + integral, into
the 1e Fock assembly.
**Lands in.** `initial_guess.f90` + a SAP-potential evaluator (reuse XC grid).
**Applies to.** Fresh runs incl. metals.

## A3. GWH — Generalized Wolfsberg–Helmholtz
**Mechanism.** F_ij = ½·K·S_ij·(H_ii + H_jj), K≈1.75; diagonalize once.
**Why for LIO.** Drop-in on the existing 1e path (we already build H and S).
Cheap, modestly better than bare core. Good warm-up / sanity lever and a
fallback when SAD/SAP data is unavailable for an element.
**Risk.** Low. **Effort.** Trivial (a few lines in `initial_guess.f90`).
**Applies to.** Fresh runs.

---

# Family B — Cross-step reuse (the big multiplier for QM/MM MD)

## B1. ASPC — Always-Stable Predictor-Corrector density extrapolation (Kolafa 2004)
**Mechanism.** At MD step t+1, predict the density from k previous *converged*
densities with the time-reversible ASPC weights, then restore N-representability
(McWeeny purification: P ← 3P²−2P³ once or twice, or Löwdin-S-orthogonalized
purification), and feed as the SCF guess.
  P_pred = Σ_{j=1..k} c_j · P(t+1−j),  c_j from the ASPC k-th order formula.
**Why for LIO.** This is the **highest aggregate-payoff strategy** because LIO's
mission is QM/MM MD. `VCINP` only reuses the *single* last density; ASPC uses the
*trajectory* and predicts forward. Typical BOMD: **~10–15 → 3–5 iters/step**.
Does **not** touch the convergence-sensitive inner loop — it only improves the
starting point, so the heme-chaos risk is contained to "did the guess land in
the right basin," which purification + a one-step fallback handles.
**Risk.** Medium. Must (a) purify to keep idempotency (else the first SCF iter is
worse, not better), (b) detect a bad extrapolation (energy jump) and fall back to
plain VCINP for that step, (c) store P history (k·M² doubles).
**float32.** Extrapolation/purification in double; XC noise unaffected.
**Effort.** Medium — P-history ring buffer + ASPC weights + purification + guard,
wired where SCF is entered per MD step (driver / `liomain` / the AMBER-GROMACS
entry).
**Lands in.** new module `lioamber/scf_extrapolation.f90`, called before the SCF
guess branch; needs access to the per-step converged `Pmat_vec` history.
**Applies to.** BOMD / geometry optimization (the production path). No effect on
single-points.

## B2. XL-BOMD — Extended-Lagrangian BOMD (Niklasson 2008+)
**Mechanism.** Carry an auxiliary density `n` propagated with time-reversible
(Verlet-like) dynamics plus a weak dissipation kernel; use `n` as the SCF guess.
The auxiliary density stays close to the SCF solution → **near 1-iter SCF** and,
crucially, **no energy drift** over long trajectories (the usual penalty of
under-converged BOMD forces).
**Why for LIO.** The end-state for fast, stable QM/MM MD. Strictly better than
ASPC at steady state, but it changes the MD *integrator* (couples to the
thermostat / needs the dissipation coefficients tuned, and forces must be
consistent with the propagated density).
**Risk.** Medium-high (dynamics correctness, not just iter count). Validate
energy conservation over long NVE runs.
**Effort.** High. **Applies to.** MD only. **Sequence after** ASPC (ASPC is the
safe stepping stone; XL-BOMD is the upgrade).

## B3. Fock / orthogonalized-density extrapolation
**Mechanism.** Same idea as ASPC but extrapolate F (or the ON-basis P′) instead
of the AO density; sometimes more stable because F is smoother across steps.
**Why for LIO.** Cheap variant once B1 infra exists; A/B test against B1.
**Risk/Effort.** Low incremental once B1 is in. **Applies to.** MD.

---

# Family C — Better accelerator algorithm (all cases, especially heme)

## C1. Staged EDIIS → bDIIS → DIIS as the default path
**Mechanism.** EDIIS (Kudin–Scuseria–Cancès 2002, energy-based, provably
convergent) in the **wild far-from-convergence** phase; switch to Pulay DIIS once
the DIIS error drops below a threshold (`DIIS_start`); optional biased-DIIS in
between. The thresholds already exist in `converger_data`.
**Why for LIO.** Directly targets heme's documented failure mode — the *wild
early-extrapolation phase* where ‖c‖~1.87 DIIS coefficients are load-bearing but
also where plain DIIS thrashes. This is the memory's explicitly-listed
**untried** lever. Make the staged handoff the *default* (today default is plain
DIIS=2) and tune the crossover.
**Risk.** Medium — changes the trajectory (validate heme median, all e2e). EDIIS
needs the per-iteration energy (extra `energ_all_iter` cost in the wild phase
only).
**float32.** EDIIS's energy comparison is in double; fine.
**Effort.** Low–medium — mostly enabling + tuning existing `converger_ediis.f90`
and the switchover logic in `converger_commons.f90`.
**Lands in.** `converger_subs/converger_commons.f90` (method dispatch),
`converger_ediis.f90`. **Applies to.** all, biggest on heme.

## C2. ADIIS — Augmented DIIS (Hu–Yang 2010)
**Mechanism.** Same wild-phase role as EDIIS but the objective uses only F and P
(no separate energy evaluation), minimizing a quadratic in the mixing
coefficients with positivity constraints.
**Why for LIO.** Cleaner than EDIIS (skips the extra energy eval), often as
robust. A drop-in alternative to C1's far-phase engine.
**Risk/Effort.** Medium. **Applies to.** all. Decide C1 vs C2 by benchmark.

## C3. Preconditioned DIIS (orbital-energy / KAIN-style)
**Mechanism.** Before extrapolating, precondition the `[F,P]` error vector by the
inverse orbital-energy gap: e_ai ← e_ai / (ε_a − ε_i) (in the MO/ON basis). This
rescales the error so small-gap (stiff) and large-gap (soft) modes are balanced.
**Why for LIO.** The conditioning cure for **small-gap systems** — exactly heme's
near-degenerate Fe d-d gap that makes EMAT near-singular (see
`heme_dgels_rank_deficiency_2026_05_28`). Improves DIIS broadly, not just heme.
**Risk.** Medium — changes the error metric → ulp-shifts the trajectory; heme is
the sensitive case, but this attacks heme's *root* conditioning problem so it may
*help* the chaos rather than worsen it. Validate on median.
**Effort.** Medium — needs ε in the DIIS error step (already have `morb_energy`).
**Lands in.** `converger_diis.f90` (`diis_fock_commut`/error construction).
**Applies to.** all, targeted at heme.

## C4. ODA — Optimal Damping Algorithm (Cancès–Le Bris 2000)
Already written up in detail in
[`heme_scf_advanced_methods_2026_05_28.md §2`](heme_scf_advanced_methods_2026_05_28.md).
**Mechanism.** Mix previous and current density along the segment with the
*analytically optimal* coefficient (minimizes the quadratic energy model),
guaranteeing monotone energy decrease far from convergence; hand off to DIIS
near convergence.
**Why for LIO.** Robust wild-phase engine; `do_rho_ls` is a cousin and could be
generalized. **Risk/Effort/Scope:** see the detailed doc. **Applies to.** all.

## C5. Modified Broyden / Anderson density mixing
Detailed in [`heme_scf_advanced_methods_2026_05_28.md §3`](heme_scf_advanced_methods_2026_05_28.md).
Quasi-Newton mixing in density space; risk noted there that it can fail the same
way as windowed DIIS on heme. Keep as an alternative accelerator, not a priority.

---

# Family D — Tame the hard-case physics (heme / transition metals)

## D1. Adaptive level shifting (re-tune the existing, dormant mechanism)
**Mechanism.** Shift virtual-orbital diagonal energies up by Δ in the ON basis
(the machinery `Shift_diag_ON` + `level_shift` already exists), damping the
density response and killing small-gap oscillations. Make Δ adaptive: fire when
`HL_gap` is small *relative to the d-d scale*, magnitude ∝ gap deficit, ramped to
0 as the DIIS error drops (so it does not perturb the final converged state).
**Why for LIO.** The mechanism is present but **never fires** (cut 0.005 ≪ heme's
0.0615 gap). This is the textbook cure for the Fe d-d oscillation. Low effort
because the apply-path exists — this is re-gating + magnitude policy.
**Risk.** Medium (trajectory) but **must converge to the unshifted state** (ramp
to 0). A prior fixed-shift attempt was rejected for being mis-gated, *not* for
being wrong in principle (see `heme_iter_chaos_diagnostics_2026_05_28`).
**Effort.** Low. **Lands in.** `converger_commons.f90` (the `lvl_shift_cut` gate
+ a schedule for `lvl_shift_en`). **Applies to.** small-gap / TM, heme.

## D2. Fractional occupation / Fermi-Dirac smearing
**Mechanism.** Occupy orbitals with f_i = 1/(1+exp((ε_i−μ)/kT)) (μ from electron
count), build the density with fractional f, anneal T→0 over the SCF. Removes the
integer-occupation discontinuity at near-degeneracy.
**Why for LIO.** Heme's chaos is substantially "which near-degenerate Fe d-orbital
is occupied" flipping iteration-to-iteration. Smearing makes the occupied set a
smooth function of ε → kills the flip. This attacks the **root cause** the memory
keeps circling. Standard for TM/metallic SCF.
**Risk.** Medium-high — must anneal to T=0 to recover the correct integer-occupied
ground state (else energy/state shifts); density build and the
energy-weighted-density path must accept fractional f; HOMO-LUMO gap definition
changes during annealing.
**float32.** Neutral. **Effort.** Medium-high — `Dens_build`/occupation handling,
μ solve, annealing schedule. **Lands in.** density build + `converger`.
**Applies to.** heme / TM. Possibly *the* heme fix if C+D1 are insufficient.

## D3. Maximum Overlap Method (MOM) / orbital following
**Mechanism.** Choose the occupied set each iteration by **maximum overlap** with
the previous iteration's occupied orbitals (max Σ|⟨ψ_i^old|ψ_j^new⟩|) instead of
strict aufbau, preventing occupied/virtual swaps across a near-degeneracy.
**Why for LIO.** Same target as D2 (the d-orbital flip) but much cheaper — just a
reordering rule on the eigenvectors we already compute. Pairs well with D1.
**Risk.** Medium — can lock onto an excited state if the guess is poor; usually
combined with aufbau for the first few iters. **Effort.** Low–medium (overlap
projection after `Diagon_datamat`). **Applies to.** heme / TM.

---

# Family E — Reach the threshold sooner (noise) + reformulation

## E1. Noise-robust convergence criterion  ← safest single win  **[DONE 2026-06-14]**
**STATUS: SHIPPED.** `converger_check` now converges on the original tight test
OR a robust-stationarity disjunct: `diis_error < conv_commut_tol(1e-4)` AND
`rho_diff < conv_rho_relax(5e-6)` AND `e_diff < conv_ediff_tol(5e-6)`. The energy
guard is essential — without it an open-shell restart (Fe3H2O6) stopped mid-descent
on a single DIIS commutator drop (ΔE still 235µHr) and missed the 150µHr e2e
tolerance by 60µHr. Provably **non-increasing** in iters (disjunction; trajectory
untouched — `converger_check` only decides *when* to stop). Results: **fosfato
25→24** (energy −2148.6510063, 26µHr from ref, within 150µHr tol); all e2e PASS
(agua/OxyMol/Fe3H2O6/ECP/QMinPcharges/fosfato + TD 05/07). Heme: **correct basin
across OMP={1,4,6,8}** (−2829.632x), iter count is run-to-run *nondeterministic*
on hybrid builds so the heme win is unmeasurable but the change cannot regress it.
The deterministic win (well-behaved closed-shell tail, ~1 iter) is modest on
single-points but recurs every converged MD step. Thresholds are module params in
`converger_data` (no env toggle).

**Mechanism.** Declare convergence on the already-computed commutator norm
‖[F,P]‖ (the DIIS error, `diis_error`) and/or a smoothed/median of `rho_diff`,
instead of the raw per-iter `rho_diff`/energy that wobble inside the float32-XC
noise band.
**Why for LIO.** Memory measured that criterion smoothing alone "shaves 5–7 iters
off the lucky-dip tail." This is the **only fully safe lever**: it changes the
*stop decision*, not the trajectory, so it cannot break heme's basin — it can only
stop the wobble earlier. It also makes a clean ruler for judging every other
method.
**Risk.** Low. **float32.** This is precisely the float32-tail mitigation that
costs nothing. **Effort.** Low. **Lands in.** `converger_check` /
`converger_commons.f90` stop test. **Applies to.** all (esp. heme tail).

## E2. Double-precision XC Fock in the final iterations only
**Mechanism.** Keep the hybrid float32 XC for the bulk of the SCF, but switch the
XC Fock contribution to double for the last ~3 iters (small ΔP, where float32
roundoff dominates the criterion), letting the SCF settle to tolD instead of
oscillating in the noise floor.
**Why for LIO.** `FULL_DOUBLE` everywhere is 2.5× slower and was rejected, but the
*tail-only* double XC is cheap and targets the documented 5–7-iter noise tail.
**Risk.** Medium — the precision switch itself perturbs the trajectory at the
switch iter; and the **double GPU-XC path is reportedly broken** (memory) → must
be fixed/validated first. **Effort.** Medium-high. **Applies to.** all, tail.
**Depends on** a working double XC kernel path.

## E3. Incremental Fock build (ΔP-driven)
**Mechanism.** Build F from the density *difference* ΔP each iteration (Coulomb +
XC of the increment), accumulating into a running F. Reduces numerical-noise
accumulation and per-iter cost.
**Why for LIO.** More a per-iter-cost + stability play than a direct count play,
but a smoother F-history can shorten the tail. **Risk.** Medium (screening
thresholds, noise bookkeeping). **Effort.** High (touches int3lu/g2g). Lower
priority.

## E4. Second-order SCF (SOSCF / QC-SCF) near convergence
Detailed as TRRH/QC-SCF in
[`heme_scf_advanced_methods_2026_05_28.md §1`](heme_scf_advanced_methods_2026_05_28.md).
**Mechanism.** Once DIIS error is small, switch to a quasi-Newton orbital-rotation
step with an approximate (diagonal) orbital Hessian → quadratic convergence,
2–3 iters to tight tolerance.
**Why for LIO.** Cuts the tail and is robust on hard cases. **Caveat:** float32
XC noise may cap the achievable tightness (the Hessian/gradient inherit the
noise). **Risk.** Medium-high. **Effort.** High. See the detailed doc.

## E5. TRAH-SCF (Trust-Region Augmented Hessian)
**Mechanism.** Full second-order method with a trust region on the augmented
Hessian eigenproblem; converges hard open-shell TM systems from poor guesses
without DIIS babysitting.
**Why for LIO.** Literally designed for heme-class problems; the most powerful
option. **Risk.** Medium-high. **Effort.** Highest (orbital Hessian, AH solve,
trust-region control). Long-term ceiling.

---

# Recommended implementation order (proposal)

Logic: build a trustworthy iter-count ruler first → bank the production
multiplier → cheap strikes at heme → heavy reformulations last. **Every
trajectory-changing item is judged on the median iter count across
OMP={1,4,6,8}, and on final-energy basin, not on the lucky default.**

| Tier | Item(s) | Rationale | Impact | Risk | Effort |
|------|---------|-----------|--------|------|--------|
| **0** | **E1** criterion on ‖[F,P]‖ + energy guard | only zero-trajectory-risk win; becomes the ruler | **DONE**: fosfato 25→24, all e2e PASS, heme basin OK | low | low |
| **1** | **A3** GWH → **A1/A2** SAD/SAP | guess-only, can't break the converged result | 25→~16 (fresh) | low | low–med |
| **2** | **B1** ASPC extrapolation | LIO is MD-first; multiplies every step; isolated from inner loop | 10–15→3–5/step | med | med |
| **3** | **D1** adaptive level-shift + **D3** MOM | mechanism exists / cheap; first strike at d-degeneracy | heme chaos | med | low |
| **4** | **C1/C2** EDIIS/ADIIS default → **C3** preconditioned DIIS | wild-phase + small-gap conditioning | broad + heme | med | med |
| **5** | **D2** smearing | root-cause heme fix if 3–4 insufficient | heme | med-hi | med-hi |
| **6** | **E4/E5** SOSCF/TRAH · **B2** XL-BOMD · **E2** double-XC tail | quadratic tail / ultimate MD / noise floor | hardest | med-hi | high |

**Start recommendation:** Tier 0 (E1) immediately — it is safe and gives the
measurement baseline. Then split: **B1 (ASPC)** for production-MD impact, and
**D1+D3** as the cheap heme first strike. Hold Family C tuning until E1's ruler
is in, and hold smearing/second-order/XL-BOMD until the cheap tiers are measured.

---

## Cross-references
- [`heme_scf_advanced_methods_2026_05_28.md`](heme_scf_advanced_methods_2026_05_28.md)
  — deep dives on ODA (C4), Broyden/Anderson (C5), QC-SCF/TRRH (E4), SAD-d (A1).
- [`heme_iter_count_chaos_diagnostics_2026_05_28.md`](heme_iter_count_chaos_diagnostics_2026_05_28.md)
  — the chaos characterization + rejected level-shift gating (informs D1).
- [`heme_diis_stability_dead_ends_2026_05_28.md`](heme_diis_stability_dead_ends_2026_05_28.md)
  — Tikhonov/|c|cap/FULL_DOUBLE rejected; trajectory Lyapunov-divergent.
- [`heme_scf_hotpath_and_sad_d_refutation_2026_05_29.md`](heme_scf_hotpath_and_sad_d_refutation_2026_05_29.md)
  — why SAD-d won't help heme-restart specifically (A1 caveat).
- Memory `initial_guess_blas_2026_06_14` — per-iter path proven near-floor; this
  catalogue is the agreed next direction.
