---
status: OPEN
date: 2026-04-25
impact: 100-200 ms (5-10% wall on fosfatoQMMM)
risk: medium
area: lioamber (typedef_operator)
---

# Diagonalization avoidance (McWeeny purification, partial diag)

## Idea

The full Fock diagonalization (`DSYEVD`) costs ~10 ms per SCF iter × 25 iters
= **250 ms wall**, called from `fock_aop%Diagon_datamat`. We don't actually
need eigenvalues for most of SCF — we only need the **density matrix** P
that corresponds to the lowest N_occ eigenvectors of F.

Two lines of attack:

### (A) McWeeny purification

Given F, the density `P_occ` is the projector onto occupied MOs:
```
P_occ = (1/2)(1 − sign(F − μI))   [μ = chemical potential mid-gap]
```
The sign function can be evaluated via McWeeny iteration:
```
P_{k+1} = 3 P_k² − 2 P_k³
```
which converges quadratically to the projector. 4-7 iterations of two GEMMs
each → ~12 GEMMs (M×M, M=86: small). Total: ~3-4 ms vs 10 ms diag.

No eigenvectors are computed during SCF; we still do **one** full diag at
the end for Mulliken/dipole/orbital energies.

### (B) Davidson partial diag

`DSYEVD` computes all M eigenpairs. We only need N_occ + a few virtuals
(for DIIS commutator). Davidson's algorithm finds the lowest k eigenpairs in
O(k · M² · n_iter) FLOPs vs O(M³) full diag. For M=86, N_occ=21, k=30: ~3×
fewer FLOPs.

cuSOLVER doesn't ship a Davidson solver, but ARPACK does (Fortran), and
custom Davidson routines are short (~200 LOC).

## Where it lives

**lioamber side**, no GPU touch.

- **`lioamber/typedef_operator/`** — `Operator` type (defined in
  `typedef_operator.f90`); methods `Diagon_datamat`,
  `BChange_AOtoON`/`BChange_ONtoAO`, `Sets_data_AO`/`Gets_data_AO`. The
  diagonalization is in `matrix_diagon_dsyevd.f90`.
- **`lioamber/SCF.f90:607,649`** — call sites: closed-shell and open-shell
  diag inside the SCF loop.
- **`lioamber/SCF.f90:624,666`** — `Dens_build` constructs P from MO
  coefficients. With McWeeny we'd skip this and directly produce P.
- **`lioamber/converger_subs.f90`** — DIIS commutator `[F,P]` requires
  both F and P in ON basis; works just as well with McWeeny-built P.

## Files to modify

For McWeeny (the higher-payoff option):

1. **New file `lioamber/typedef_operator/matrix_purify_mcweeny.f90`**:
   - Subroutine `purify_mcweeny(F_on, NCO, P_on, max_iter, tol)`.
   - Initial guess: `P_0 = (1/2)(1 − F̃)` where `F̃ = (F − (Hmax+Lmin)/2 · I)/scale`,
     with `Hmax,Lmin` estimated from prior iter's eigenvalues (cached).
   - Loop: `P ← 3P² − 2P³` via two DGEMM calls per iter; check
     `‖P − P²‖_F < tol`.
   - Output: `P_on` (idempotent projector in ON basis).

2. **`lioamber/typedef_operator/typedef_operator.f90`**:
   - Add `Operator::Purify_to_density(NCO, P_on)` method that calls
     the new routine.

3. **`lioamber/SCF.f90`**:
   - Inside SCF loop, replace the diag + `Dens_build` block:
     ```fortran
     call fock_aop%Purify_to_density(NCOa_f, P_on)
     call rho_aop%Sets_data_ON(P_on)
     call rho_aop%BChange_ONtoAO(Ymat, M_f)   ! for grid Vxc
     ```
   - On the **last 2 iters** (`good < told × 10`), do a regular diag to
     produce eigenvalues for DIIS error vector and final orbital
     properties. Add a flag `use_purification` (default OFF, namelist).
   - **After** SCF converges, do one final full diag for Mulliken / dipole
     / orbital energies (already done for `Eorbs` at line 628).

4. **`lioamber/init_lio.f90`** — namelist option `use_purification`,
   `purification_max_iter` (default 10), `purification_tol` (1e-9).

## Math correctness

McWeeny purification is mathematically exact for a converged Fock matrix:
the fixed point of `3P² − 2P³` is the spectral projector onto the occupied
subspace. Quadratic convergence: error halves twice per iter near the
fixed point.

**Open-shell**: applies independently to α and β projectors → 2× the
cost (~6-8 ms). Net savings vs 2 diags (20 ms): still ~12-15 ms / iter.

**Convergence prerequisite**: F must have a clear HOMO-LUMO gap. For
metallic / near-degenerate systems, McWeeny can fail. Detection: monitor
`‖P − P²‖_F`; if not converging in `max_iter`, fall back to diag.

## Why I trust this works

McWeeny is the **standard** linear-scaling DFT kernel (CONQUEST, ONETEP,
SIESTA's order-N mode). For dense small-M cases it's "merely faster than
diag", not asymptotically better. Empirically:
- M=86: 4-5 GEMMs of 86×86 ~ 0.5 ms each = 2.5 ms vs DSYEVD 10 ms.
- M=200: 4-5 GEMMs of 200×200 ~ 3 ms each = 15 ms vs DSYEVD 50 ms.

Always wins, especially as M grows.

## Validation

1. **Unit**: `test/unit/lioamber/test_purify_mcweeny.f90` — random Fock,
   purify, verify `P² = P` to 1e-10 and `Tr(P) = N_occ` exactly.
2. **E2E**: `./run_tests.py` with `use_purification=.true.`. All 30 tests;
   energies within `1e-6 Ha`; iter count ±2.
3. **Open-shell**: 02_Fe3H2O6 (Fe3+ HS, near-degenerate d-orbitals) — most
   likely to expose McWeeny stalling. May need fallback.
4. **TDDFT**: 07_TDDFTHCL — only ground state SCF uses purification;
   verify it doesn't pollute the post-SCF orbital eigenvectors used by
   the propagator.
5. **Density-fitting integration check**: P comes from purification with no
   diagonalization; must be valid input to int3lu (which only reads
   `rho` packed). Verify `rho` from purified P matches diag-built P
   to `1e-9` per element.
6. **Reproducibility**: `./run_unit.py --sanitize=racecheck` (CPU-only
   for Fortran code).

## What could kill this

- **Near-degeneracy systems** (transition-metal, multi-reference) where
  HOMO-LUMO gap is small — McWeeny stalls or oscillates. Mitigation:
  detect and fall back to diag. Cost: a few ms wasted on the failed
  purification attempt.
- **DIIS error vector quality**: the commutator `[F,P]` quality depends
  on having a well-converged P. Purified P is *exactly* idempotent,
  which is actually **better** than diag-built P (which has float
  rounding). May actually help DIIS in float32.
- The conditional structure (purify mid-SCF, diag at end) needs careful
  bookkeeping; `Eorbs` and `MO_coef_at` aren't available mid-SCF for
  any caller that needs them. Audit any other consumer of these arrays
  inside the loop.

## Davidson alternative

If McWeeny fails on too many systems, ARPACK Davidson is plan B:
- Pros: uses cuBLAS-friendly `cublasDsymv`, can request only N_occ + 5
  eigenpairs.
- Cons: more code, library dependency, less elegant.
- Defer unless McWeeny proves unviable.

## Recommendation

Implement McWeeny first behind a flag. Run on 30-test suite. If ≥27
pass without iter regression, enable by default with diag fallback.
This is the cleanest "outside the box" lever — it removes O(N³) work
from the inner loop entirely.
