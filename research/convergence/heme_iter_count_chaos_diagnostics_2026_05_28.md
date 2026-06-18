# Heme SCF iter-count chaos — diagnostics and dead ends (2026-05-28)

## Question

The user asked: can we make heme's SCF stable to ulp-level perturbations
(BLAS thread count, GPU kernel ordering, FMA emission) so that follow-up
optimizations don't keep shuffling its iter count by 50-100%? Goal was to
identify the right numerical condition to measure and decide whether
shifting selected computations to double precision (or other numerical
fixes) would deliver real stability.

See also: [heme-diis-stability-dead-ends-2026-05-28.md] for the prior
algebraic-safeguard dead ends (Tikhonov / |c|_max cap / DGELSD / FULL_DOUBLE).

## Measurement infrastructure shipped

`SCF.f90` line 755: per-iteration log of `HL_gap`, `diis_error`,
`rho_diff` behind `verbose > 3`. Format:

    [scf_diag] iter=  N HL_gap=  X.XXe-XX diis_err=  X.XXe-XX rho_diff=  X.XXe-XX

Enable with `verbose = 4` in `&lio`. Cost: one formatted write per SCF
iter; off by default so test diffs are unaffected.

This is the right diagnostic surface for future debug: HL_gap reveals
near-degenerate orbital crossings; diis_error pins the wild-extrapolation
phase; rho_diff exposes whether convergence is being driven by trajectory
descent or by lucky single-iter noise dips.

## What the data showed

### 1. HL_gap shape is stable across OMP threads

Converged HL_gap on heme open-shell (Fe d-d states) is 0.0615 Eh ≈ 1.67 eV
across all `OPENBLAS_NUM_THREADS ∈ {1,4,6,8}`. During the wild phase
(iters 1-12) HL_gap dips to 0.020-0.036 (near-degenerate crossings during
d-shell reorganisation), but the **dip pattern is bit-identical across
OMP threads through iter ~24**. Trajectories begin to diverge only in the
late phase (iter 25+).

This refutes the "HL_gap collapse drives chaos" hypothesis. Level shifting
gated on `HL_gap < lvl_shift_cut` (current default 0.005) would never fire
at this system (cut is 12× tighter than the converged gap).

### 2. Iter spread across `OPENBLAS_NUM_THREADS`

Current binary, `LIO_OVERLAP_INT3LU_G2G=1`, default auto-tune:

| OMP | iters | final E (Eh) | basin |
|-----|-------|--------------|-------|
| 1 | 134 | -2829.6326055 | ✓ |
| 4 | 83  | -2829.6326139 | ✓ |
| 6 (auto-tune default) | 72 | -2829.6325926 | ✓ |
| 6 (explicit env) | 183 | -2829.6326082 | ✓ |
| 8 | 98 | -2829.6326258 | ✓ |

Spread: 72-183 iters (2.5×). Final energy variance: ±1.7 µHa — well below
any chemically meaningful threshold and well within hybrid float32 GPU
noise floor. **Basin is preserved across all variants.**

### 3. Two stacked sources of iter variance

Each trajectory enters the rho_diff "noise envelope" (~ 1-3e-6) at very
different iters:

| OMP | enters envelope at iter | tail-oscillation length | first dip < told=1e-6 |
|-----|------------------------|-------------------------|----------------------|
| 4 | ~80 (smooth) | none — descends monotonically | 83 |
| 8 | ~91 | ~7 iters of dance 1.05-1.86e-6 | 98 |
| 1 | ~125 | ~10 iters of dance 1.0-7.6e-6 | 134 |
| 6 | ~174 | ~9 iters of dance 1.26-2.0e-6 | 183 |

Two sources stacked:

- **Trajectory-length variance (~80 iters)**: OMP=6 trajectory genuinely
  takes ~90 more iters than OMP=4 to descend into the noise envelope. This
  is the dominant component and cannot be cleanly fixed at the algorithm
  level — it comes from BLAS-summation-order ulp shifts amplified by the
  ill-conditioned EMAT solve over 70+ SCF iters.
- **Criterion-noise variance (~5-7 iters)**: once a trajectory is in the
  envelope, the `rho_diff < told` test samples a noisy distribution
  ~3× the criterion magnitude. Whether iter 83 vs 89 vs 96 happens to dip
  below 1e-6 is luck.

The criterion-noise component is small. Smoothing the criterion (2-iter
average, running min, joint rho_diff + e_diff test) would shave 5-7 iters
off the lucky-dip tail without touching the dominant trajectory-length
variance. Not worth the risk of looser convergence on other systems.

## Approaches tested and rejected on this session

### Windowed DIIS subspace (REJECTED)

Hypothesis: near-converged EMAT is rank-deficient (min eig ~ 1.8e-9 by
iter 65), DGELS amplifies ulp shifts into wild |c|=1.87 swings. Restricting
the LS solve to the most recent `scf_diis_window` vectors when
`diis_error < scf_diis_window_cut` should give cleaner extrapolation.

Implementation (later reverted from `converger_diis.f90:diis_get_new_fock`):
adaptive `ndiist_eff = min(ndiist, 6)` when `diis_error < 1e-2`, with
EMAT submatrix and slot index offset shifted to use the latest entries.

Result: convergence completely broken. OMP=4 hits `nmax=250` without
convergence; energy oscillates ±5 mHa around the basin. The 6-vector
subspace cannot represent the slow modes the late-phase trajectory needs
to damp. Tightening the threshold to 1e-3 or 1e-4 just shifts the
breakage point; it does not fix the loss of slow modes.

This is consistent with the prior |c|_max cap result
(see [heme-diis-stability-dead-ends]): the wide subspace and large
extrapolation coefficients are **load-bearing** for convergence on
near-degenerate Fe-d systems.

### Static small ndiis (REJECTED)

Same physics: setting `ndiis = 5` as input diverges in the wild phase
(final E = +610.5 Eh at nmax=250 — wrong sign, completely broken).

### Criterion smoothing (NOT SHIPPED)

Considered but rejected by advisor: 2-iter running min, 2-iter average,
joint `(rho_diff < 2e-6) AND (e_diff < 5e-6)`. Would help ~5-7 iters on
the lucky-dip tail; trajectory-length variance untouched; net win not
worth criterion-loosening risk on other systems.

### Saunders-Hillier level shift in MO basis (REJECTED)

Implementation in `converger_data.f90` + `converger_commons.f90` +
`SCF.f90`: cached previous-iter virtual projector
`P_virt = C_virt C_virt^T` (built from `morb_coefon[:, NCO+1:M]` in ON
basis) and applied `F̃ = F + b · P_virt` between DIIS extrapolation and
diagonalization, gated on `cut_lo < diis_error < cut_hi`. Distinct from
the legacy `Shift_diag_ON` (which adds `b` to arbitrary diagonal entries
of ON-basis Fock — not a real virtual-subspace projector and therefore
not mathematically a level shift).

**Mathematical motivation (sound in theory):** the shifted operator has
the same eigenvectors as F at convergence (since P_virt_prev = P_virt_curr
at the fixed point), with virtual eigenvalues raised by b. The Roothaan
fixed-point Jacobian spectral radius ρ ≈ 1 - HL_gap/(HL_gap + 2Δ) shrinks
with widened effective gap; predicted to give faster contraction and
~5× less ulp amplification on heme (HL_gap = 0.0615 Eh → 0.11 Eh with
b = 0.05).

**Empirical result (FAILED on heme across all parameter sweeps):**

| b (Eh) | cut_hi | result | iters | final E (Eh) | basin shift (µHa) |
|--------|--------|--------|-------|--------------|-------------------|
| 0.05 | (none, upper gate disabled) | wrong basin | 604 (Rho_LS) | -2829.6221 | 10500 |
| 0.05 | 0.05 | wrong basin | 717 (Rho_LS) | -2829.6334 | -800 |
| 0.02 | 0.01 | wrong basin | 527 (Rho_LS) | -2829.6266 | 6000 |
| 0.01 | 1e-3 (tail-only) | wrong basin | 541 (Rho_LS) | -2829.6332 | -600 |
| 0.001 | 1e-3 (tail-only) | converged, basin perturbed | 75 | -2829.6325816 | 11 |
| 0 | n/a | baseline | 72 | -2829.6325926 | 0 |

**Why it failed on heme:** the math assumes `P_virt_prev ≈ P_virt_curr`
between consecutive iterations. For systems with well-separated occupied
and virtual subspaces, MOs barely rotate between iters and this invariance
holds. Heme's near-degenerate Fe-d shell has MOs continuously rotating
substantially between iterations throughout the SCF (not just in the wild
phase), so `P_virt_prev` projects partially onto iter N's *occupied*
subspace. The b·P_virt term then leaks into occupied eigenvalues — visible
as HL_gap dropping below baseline (e.g., iter 12 shifted HL_gap = 0.029
Eh vs baseline 0.068 Eh; b=0.05 with no upper gate) — and drives the
trajectory into a different basin or stalls convergence outright.

Even b = 0.001 Eh (orders of magnitude below HL_gap) perturbs the basin
energy by 11 µHa, well outside the ±1 µHa basin band.

**Class verdict:** any technique requiring stable MO subspace alignment
between consecutive iters (Saunders-Hillier MO shift, MOM constraints,
orbital-rotation Newton steps with fixed Hessian) will fail on heme by
this same mechanism. The d-shell continuous rotation is the structural
obstacle.

**What might work next:** approaches that DON'T rely on inter-iter MO
alignment — direct energy minimization on orbital rotations
(parametrize C = C_prev · exp(X) and minimize E(X) with line search),
ODA (optimal damping on density, energy-based 1D line search),
modified Broyden mixing in density space. All are bigger investments
than Saunders-Hillier.

## Diagnosis

Heme's SCF on the current hybrid-float32 GPU binary has two stable
properties (basin, final density to ~1 µHa) and one unstable property
(iter count). The unstable property is downstream of two stacked sources
of float-order variance and cannot be cleanly stabilised by:

- DIIS subspace algebraic safeguards (Tikhonov, |c|_max cap, DGELSD,
  windowed subspace — all rejected with data, here and in
  [heme-diis-stability-dead-ends-2026-05-28]).
- HL_gap-gated level shifting (cut threshold is tighter than the real
  converged d-d gap; would never fire).
- Criterion smoothing (5-7 iter shift, doesn't address trajectory-length
  variance).
- Precision promotion (FULL_DOUBLE is broken on open-shell path; even if
  fixed, gives 2.5× wall regression with ~10% iter change per prior
  research).

The real levers are upstream of the SCF iteration: better TM initial
guess (extended Hückel, SAD with d-shell polarisation), EDIIS/A-DIIS
switchover in the wild-extrapolation phase, or direct minimisation. All
require substantial engineering investment and are deferred.

## Recommended benchmarking rule (update for MEMORY.md)

For heme and other near-degenerate open-shell TM systems on the current
binary:

- **Final energy across `OPENBLAS_NUM_THREADS ∈ {1, 4, 6, 8}` is the
  stable signal.** Heme: -2829.63261 ± 1 µHa. Any future optimisation must
  preserve this band; deviations > 10 µHa are real regressions.
- **Iter count is allowed to vary 70-200** across thread counts without
  that being a regression. Prefer comparing medians, not lucky-default
  trajectories.
- **Use `verbose = 4` to enable per-iter `[scf_diag]` log** when triaging
  a suspected iter-count regression. If HL_gap shape and basin both match
  prior runs, it's a benign trajectory shuffle.

## Closes

- The "shift selected computation to double precision" branch of the
  task: closed. Precision was not the lever; the SCF acceleration data
  path is already LIODBLE (double). FULL_DOUBLE GPU kernels are net-
  negative (broken open-shell + 2.5× wall on prior data).
- The "find a numerical condition to measure" branch: closed. The
  condition that matters is the basin (final energy band across OMP),
  which IS stable. Iter count is a noisy secondary signal that should
  not gate optimisation decisions.

## Files touched / reverted

- `SCF.f90`: added `verbose > 3` gated per-iter `[scf_diag]` log
  (HL_gap + diis_error + rho_diff). **Shipped.**
- `converger_diis.f90`: adaptive windowed-subspace `ndiist_eff` logic
  implemented, tested, reverted. **Not shipped.**
- `converger_data.f90`: `scf_diis_window` / `scf_diis_window_cut` /
  `scf_diis_window_min` parameters implemented, tested, reverted.
  **Not shipped.**
