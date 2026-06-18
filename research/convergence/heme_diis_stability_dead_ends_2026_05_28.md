# Heme DIIS stability — algebraic safeguards rejected (2026-05-28)

## Question

The 72-iter heme trajectory is fragile under any ulp-level upstream change
(BLAS thread count, summation order in `update_emat`, DGELS→DGELSD). User
asked: can we make this stable so follow-up optimizations don't keep
breaking heme?

## Diagnostic profile (current binary, `LIO_OVERLAP_INT3LU_G2G=1`)

Per-iteration EMAT spectrum + |c|_max via instrumented `diis_get_new_fock`:

| iter range | max\|eig\| | min\|eig\| | \|c\|_max | regime |
|------------|----------|----------|----------|--------|
| 3-15 | 1-44 | 0.05-0.6 | 0.3-0.7 | healthy DIIS |
| 17-21 | 1-45 | 2e-4 | **1.87** | wild extrapolation |
| 65-72 | 1e-6 | 1.8e-9 | 0.2 | near convergence |

EMAT is negative semi-definite (`EMAT(i,j) = tr(e_i e_j)` with anti-Hermitian
commutators `e_i = FP-PF`). The "rank deficiency near convergence" hypothesis
(emin = 1.8e-9 by iter 65) is real but is not the chaos driver — see below.

## Chaos signal on current binary (no changes)

```
OMP=OPENBLAS=1   →  134 iters, E = -2829.6326055
OMP=OPENBLAS=4   →   83 iters, E = -2829.6326139
OMP=OPENBLAS=6   →   72 iters (default auto-tune)
OMP=OPENBLAS=8   →   98 iters, E = -2829.6326258
```

Same convergence basin, ±50% iter spread. Same chaos pattern persists with
the energy-rejection rollback disabled (`scf_prev_was_diis` gate forced
.false.): 122/130/197 iters — rollback is *not* load-bearing for the
trajectory's basin stability; it shaves ~30% off the iter count on average.

## What was tried

### 1. Tikhonov regularization on the augmented Lagrangian (REJECTED)

Subtract `λ * I` from the B-block diagonal before DGELS solve, with
`λ = τ * max|diag(EMAT)|`. Aim: make the LS solve unique and dampen the
near-singular-direction amplification of ulp shifts.

Sweep across τ × OMP threads (single full heme run each):

| τ \ OMP | 1 | 4 | 8 | basin |
|---------|---|---|---|-------|
| 0 (baseline DGELS) | 134 | 83 | 98 | correct (-2829.6326) |
| 1e-5 | 94 | 162 | 127 | correct |
| 1e-4 | **543** | 116 | 291 | **wrong at OMP=1** (-2829.6246) |
| 1e-3 | 532 | 540 | 587 | wrong (~-2829.630±) |
| 1e-2 | 530 | 536 | 545 | wrong |

- τ ≤ 1e-5: same chaos pattern, spread similar or worse than baseline.
  Tikhonov at this magnitude is below the noise floor it would need to
  swamp — the smallest eigenvalues of EMAT (down to 1.8e-9 at iter 65)
  are well above ulp noise (~2e-21 absolute), so adding λ ~ 10⁻⁹ does
  almost nothing.
- τ ≥ 1e-3: SCF lands in a wrong DFT state. Heavy regularization
  destroys the legitimate extrapolation directions DIIS needs to find
  the correct minimum.
- τ = 1e-4: walks the edge — sometimes converges, sometimes catastrophic.
  Conclusion: there is no τ that simultaneously (a) tames the chaos and
  (b) preserves the correct DFT basin. Dead end.

The advisor's framing made this predictable: "Tikhonov, SVD truncation
and minimum-norm all live on the same continuum. They differ only in how
aggressively they kill the wild-extrapolation direction." On heme, that
wild direction is *load-bearing* for convergence.

### 2. |c|_max preventive cap (REJECTED — worse than baseline)

When `maxval(abs(bcoef)) > 2.0`, replace the extrapolation with
"newest-stored Fock only" (`bcoef = (0,...,0,1)`), equivalent to skipping
DIIS this iter without disturbing the EMAT/fockm history.

Heme result:

| OMP | iters | final E | basin |
|-----|-------|---------|-------|
| 1 | 461 | -2829.6326 | correct (slow) |
| 4 | 555 | -2829.6301 | **wrong** |
| 8 | 546 | -2829.6219 | **wrong** |

The wild `|c|=1.87` extrapolations at iters 18-21 (and a handful later)
are doing real corrective work. Replacing them with "no extrapolation"
strands the trajectory.

### 3. DGELSD min-norm solve (previously rejected, 2026-05-28 earlier)

Documented in `heme_dgels_rank_deficiency_2026_05_28.md`: 72 → 164 iters.
Same family as Tikhonov τ → ∞.

### 4. FULL_DOUBLE (`precision=1`) — broken, not testable

Builds successfully but heme aborts at startup with
`CudaMatrix<double>::zero — assertion this->data failed` in `matrix.cpp:293`.
Separate latent bug in the open-shell `precision=1` path. Per
`MEMORY.md`, even on a working FULL_DOUBLE build fosfato sees only 25→27
iters at 2.5× wall — not a viable ship lever as a stability fix.

## Diagnosis

The heme SCF trajectory is **Lyapunov-divergent in iter count**:
ulp-level perturbations to the data path (commutator DGEMMs in
`Commut_data_r`, summation order in `update_emat`, DIIS-solver choice)
propagate via `bcoef → fock → density → next EMAT` over 70+ iters and
amplify into 30-90% iter-count swings. The convergence basin is
preserved; only the trajectory length varies.

**No DIIS-level algebraic safeguard can fix this** because the wild
extrapolation directions DIIS exploits near a near-degenerate Fe-d state
are themselves load-bearing for convergence. Killing them (Tikhonov, |c|
cap, min-norm) collapses iter count or, worse, drops out of basin.

The existing energy-rejection rollback (138→72) is a reactive safety net
that catches catastrophic post-DIIS energy rises — that is the right
*shape* of intervention and it ships. Strengthening it further (lower
ΔE gate, tighter rho_diff gate) would make it fire on legitimate
turbulence in other systems; we did not find a tighter setting that
helps without regression risk.

## What might actually move the needle (not attempted here)

- **Pre-DIIS conditioning**: better initial guess for transition metals
  (extended Hückel, atomic SAD with d-shell polarization). The chaos
  seeds are at iters 3-4 (catastrophic +26 Eh swing); a better guess
  might avoid them entirely.
- **Direct minimization**: CG or trust-region BFGS on the energy
  functional. No DIIS, no chaos, but a large engineering investment.
- **EDIIS/A-DIIS hybrid in the wild-extrapolation phase**: EDIIS uses
  energy as the constraint, has no rank-deficient B issue. Switching to
  EDIIS while |c|_max > some threshold, then back to CDIIS for tail
  convergence, is the standard QChem/Gaussian recipe. Non-trivial to
  wire up but plausible.
- **Level shifting for the d-d gap**: shift virtual orbitals up during
  the transient phase to widen the HOMO-LUMO gap. Disabled in the
  current DIIS path (`(.not. diis_on)` gate in `converger_fock`); would
  need to be re-enabled selectively.

## Recommendation

Treat the 72-iter heme trajectory as "fast and fragile, not the spec":
- Don't gate future optimizations on preserving exactly 72 iters.
- Use median across `OPENBLAS_NUM_THREADS={1,4,6,8}` (≈100 iters) as
  the comparison metric. Any kernel/BLAS change should be evaluated
  against that, not the lucky-default 72.
- Accept that some changes will shuffle iter count by ±50% without
  fixing or breaking heme; only basin-flip or `nmax`-hit is a real
  regression.
