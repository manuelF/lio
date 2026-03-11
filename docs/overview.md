# SCF Loop Architecture & DIIS Convergence Analysis

## Overview

This document traces the full data path from GPU kernel output through DIIS
convergence, and analyzes why GPU-side float32 precision changes (e.g. Kahan
compensated summation) cause SCF convergence regressions.

---

## 1. SCF Iteration Data Flow (Closed-Shell)

Each SCF iteration in `lioamber/SCF.f90` follows this pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│ Fmat_vec = Hmat_vec  (1-electron integrals, real*8 packed)     │
│ + int3lu(Coulomb)    (density fitting, real*8 packed)          │
│ + g2g_solve_groups   (XC, GPU float32 → cast to real*8 packed) │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ spunpack('L', M, Fmat_vec, fock_a0)                   │
│   Packed lower-tri (MM elements) → Dense M×M (real*8) │
│   Off-diag: no division (Fock uses raw packed values)  │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ fock_aop%Sets_data_AO(fock_a0)                        │
│ rho_aop contains density from previous iteration      │
│                                                       │
│ conver(niter, good, good_cut, M, rho_aop, fock_aop,  │
│        Xmat, Ymat, spin=1)                            │
│                                                       │
│ Inside conver:                                        │
│   1. Extract AO-basis Fock and density                │
│   2. Transform to ON basis: F'=X^T·F·X, P'=Y^T·P·Y  │
│   3. Store F' in fockm history buffer                 │
│   4. Compute [F',P'] commutator → store in FP_PFm    │
│   5. If damping: F_new = (F + α·F_old)/(1+α)         │
│   6. If DIIS: solve EMAT·c=b, build F=Σ c_k·F'_k    │
│   7. Result: updated Fock in ON basis                 │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ fock_aop%Diagon_datamat(morb_coefon, morb_energy)     │
│   LAPACK dsyev/dgeev (real*8)                         │
│   → MO coefficients in ON basis + orbital energies    │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ morb_coefat = Xmat · morb_coefon                      │
│   Back-transform to AO basis (real*8 matmul)          │
│                                                       │
│ rho_aop%Dens_build(M, NCO, ocup, morb_coefat)        │
│   P_ij = 2·Σ_k c_ik·c_jk  (over occupied MOs)       │
│   All real*8 arithmetic                               │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Convergence check (SCF.f90:722-731):                  │
│                                                       │
│   good = 0                                            │
│   do jj=1,M ; do kk=jj,M                             │
│     del = xnano(jj,kk) - Pmat_vec_old(jj,kk)        │
│     del = del * sqrt(2)                               │
│     good = good + del²                                │
│   enddo ; enddo                                       │
│   good = sqrt(good) / M                               │
│                                                       │
│ Loop continues while good ≥ told OR Egood ≥ Etold     │
│ (told typically 1e-6, Etold typically 1e-6)           │
└───────────────────────────────────────────────────────┘
```

### Key files in the pipeline

| File | Role |
|---|---|
| `lioamber/SCF.f90` | Main SCF loop, convergence check, orchestration |
| `lioamber/converger_subs.f90` | DIIS/damping implementation |
| `lioamber/converger_data.f90` | DIIS state (history buffers, coefficients) |
| `lioamber/typedef_operator/` | Operator type: stores matrices, does base changes |
| `lioamber/packed_storage.f90` | Pack/unpack between triangular vector and dense matrix |
| `g2g/init.cpp` | `g2g_solve_groups_` entry point, dispatches to iteration |
| `g2g/partition.cpp` | `add_rmm_output`: scatters GPU float32 → host double |
| `g2g/cuda/iteration.cu` | GPU iteration driver, includes all kernels |

---

## 2. Precision Along the Path

| Stage | Precision | Location |
|---|---|---|
| GPU kernel computation | **float32** | `g2g/cuda/kernels/energy.h`, `rmm.h` |
| Scatter to host | **float32 → double cast** | `partition.cpp:add_rmm_output` |
| Accumulation into Fmat_vec | **double (+=)** | `partition.cpp:add_rmm_output` |
| 1-electron + Coulomb integrals | **double** | `int1`, `int3lu` in lioamber |
| Unpack to dense matrix | **double** | `packed_storage.f90` |
| DIIS: base change, commutator | **double** (DGEMM) | `converger_subs.f90`, `typedef_operator/` |
| DIIS: EMAT build (trace products) | **double** | `converger_subs.f90:trace_product` |
| DIIS: solve EMAT·c=b | **double** (DGELS) | `converger_subs.f90` |
| Fock diagonalization | **double** (LAPACK) | `typedef_operator/Diagon_datamat` |
| Density construction | **double** | `typedef_operator/Dens_build` |
| Convergence criterion | **double** | `SCF.f90:722-731` |

**Everything after the GPU→host cast is double precision.** The float32 noise
from GPU kernels propagates as exact double values through the rest of the
pipeline — the noise magnitude is ~1e-7 relative (float32 precision), but it's
carried without further loss.

---

## 3. The DIIS Mechanism in Detail

### 3.1 Error Matrix Construction

The DIIS error vector for iteration `i` is the commutator in the orthonormal basis:

```
e_i = [F'_i, P'_i] = F'·P' - P'·F'
```

where `F' = X^T·F·X` and `P' = Y^T·P·Y`.

The error matrix EMAT has entries:

```
EMAT(i,j) = Tr(e_i · e_j)
```

computed via `trace_product(FP_PFm(:,:,slot_i), FP_PFm(:,:,slot_j), M)`.

### 3.2 DIIS Linear System

The system solved is:

```
| EMAT  -1 |   | c   |   | 0  |
| -1^T   0 | · | λ   | = | -1 |
```

This is solved via LAPACK `DGELS` (least-squares QR factorization).
The constraint `Σ c_k = 1` is enforced by the Lagrange multiplier row.

### 3.3 Fock Extrapolation

The new Fock matrix in ON basis is:

```
F'_new = Σ_k c_k · F'_k
```

where `F'_k` are stored in the `fockm` circular buffer.

### 3.4 Damping-to-DIIS Transition

With `conver_criter=2` (standard DIIS mode), the switch is:
- Iterations 1-2: damping only (`F = (F_new + α·F_old)/(1+α)`)
- Iterations 3+: DIIS extrapolation (damping disabled)

The transition is **abrupt** — no blending between damping and DIIS.

---

## 4. Why Kahan Summation in GPU Kernels Worsened Convergence

### 4.1 What Was Tried

| Change | Effect |
|---|---|
| Kahan in `gpu_compute_density` bj-loop | 25 → 31 SCF iters (fosfatoQMMM) |
| Kahan + `__launch_bounds__(64,16)` | 25 → 31 iters, amplified |
| Kahan in `gpu_update_rmm` inner loop | Open-shell energy error 0.004 Ha |
| Reducing `ndiis` from 30 to 8 | Convergence stalls at ~2e-6 |

### 4.2 Root Cause: DIIS Sensitivity Amplification

The core issue is a **noise amplification chain**:

```
GPU float32 noise (~1e-7 relative)
    │
    ▼
XC contribution to Fock matrix (small component of total Fock)
    │
    ▼
Commutator [F', P'] near convergence:
    F' and P' nearly commute → [F',P'] ≈ 0 + noise
    The noise in [F',P'] is DOMINATED by float32 XC noise
    │
    ▼
EMAT(i,j) = Tr(e_i · e_j):
    Near convergence, EMAT entries are ~O(noise²) ≈ O(1e-14)
    The RATIO of entries determines DIIS coefficients
    Small absolute changes → large ratio changes
    │
    ▼
DIIS coefficients c_k:
    Can be large positive/negative (e.g. [5.2, -3.1, -2.8, 1.7])
    Constraint: Σ c_k = 1, but individual |c_k| >> 1
    │
    ▼
Extrapolated Fock F' = Σ c_k · F'_k:
    Large |c_k| AMPLIFY the float32 noise in stored F'_k
    Net noise in F' >> noise in any individual F'_k
```

**The critical insight**: Kahan summation doesn't just change the *magnitude*
of float32 noise — it changes the *pattern* (which bits are noisy). The DIIS
trajectory depends on the specific noise pattern, not just its magnitude. A
"more precise" pattern is not necessarily better for DIIS — it's just
*different*, and the baseline pattern happened to produce a favorable trajectory.

### 4.3 Why ndiis=30 Is Needed (Not the Typical 6-12)

With float32 noise floor ~2e-6 in `rho_diff`, the density can't converge
smoothly below this level. DIIS must find a linear combination of stored Fock
matrices that *accidentally* produces a density with `rho_diff < 1e-6`. With
only 6-12 history vectors, the search space is too small. With 30 vectors,
there are enough degrees of freedom that DIIS occasionally finds a favorable
combination.

### 4.4 Quantifying the Amplification

For fosfatoQMMM (M≈86, closed-shell GGA):

- XC Fock contribution: ~10-30% of total Fock matrix elements
- Float32 relative precision: ~1.2e-7
- Absolute XC noise per element: ~1e-7 × element_value ≈ 1e-8 to 1e-6
- Commutator near convergence: dominated by this noise
- EMAT entries: ~1e-14 to 1e-12 (noise² products)
- EMAT condition number: very large (1e6+)
- DIIS coefficients: |c_k| can reach 5-10
- Amplified noise in extrapolated Fock: 5-10× single-iteration noise

---

## 5. Hypotheses for Fortran-Side Improvements

### H1: DGELS vs DGESV for DIIS Solve

**Current**: `DGELS` (QR least-squares) solves the augmented system.
**Issue**: DGELS minimizes ||Ax-b||₂, which is overkill for a square system.
For ill-conditioned EMAT, QR and LU give different residuals.
**Alternative**: `DGESV` (LU with partial pivoting) — direct solve, may be
more stable for the exact-dimension case.

### H2: DIIS Coefficient Regularization

**Issue**: Large |c_k| amplify noise. Some DIIS implementations:
- Cap max |c_k| (e.g. Pulay's ADIIS/EDIIS hybrid)
- Add Tikhonov regularization: `EMAT + λI` before solving
- Use CROP-DIIS (constrained residual optimization)

**Trade-off**: Regularization slows convergence in the smooth regime but
stabilizes the noisy near-convergence regime.

### H3: Improved Damping-DIIS Transition

**Issue**: Abrupt switch at iteration 3 (conver_criter=2).
**Alternative**: Blend damping and DIIS for a few iterations:
```
F = α·F_DIIS + (1-α)·F_damped, α ramps from 0→1 over iterations 3-5
```

### H4: EMAT Conditioning Check

**Issue**: Near convergence, EMAT becomes nearly singular.
**Improvement**: Check condition number; if too large, fall back to damping
or use fewer history vectors for that iteration.

### H5: Double-Precision Accumulation in GPU Kernels (Safe Approach)

Instead of Kahan (which changes float32 bit patterns), accumulate the
warp-reduction result into a **double-precision** shared memory word:
```cuda
// After float32 warp reduction:
atomicAdd(&result_double, (double)warp_sum);
```
This preserves the float32 computation order (keeping DIIS trajectory
unchanged) but reduces cross-point-group accumulation noise.

**Note**: This would only help if the per-group float32 values are preserved
exactly and only the host-side accumulation order changes. Since `add_rmm_output`
already accumulates in double, this may have no effect.

### H6: Fock Matrix Zeroing

**Observation**: `Fmat_vec` is NOT explicitly zeroed at the start of each SCF
iteration. Instead, `int3lu` (Coulomb) overwrites it: `Fmat_vec = Hmat_vec +
Coulomb_terms`, then `g2g_solve_groups` ADDS XC contributions. This is correct
because `int3lu` sets `Fmat_vec = Hmat_vec + J`, then g2g adds `+XC`.

If there were ever a code path where `Fmat_vec` was not fully overwritten by
`int3lu` before `g2g_solve_groups` adds to it, stale values could accumulate.
Worth verifying.

---

## 6. Fock Matrix Assembly: Detailed Breakdown

The Fock matrix is built from three components each iteration:

```fortran
! SCF.f90, inside the SCF loop:

! 1. Coulomb + 1-electron (int3lu overwrites Fmat_vec)
call int3lu(E2, Pmat_vec, Fmat_vec2, Fmat_vec, Gmat_vec, Ginv_vec, &
            Hmat_vec, open, MEMO)
! After this: Fmat_vec = H_core + J (Coulomb) — all real*8

! 2. XC (g2g ADDS to Fmat_vec)
call g2g_solve_groups(0, Ex, 0)
! After this: Fmat_vec = H_core + J + XC
! The XC part came from GPU float32, cast to double, accumulated

! 3. (Optional) Reaction field, external field additions
```

The XC contribution is typically 10-30% of the total Fock matrix. The float32
noise affects only the XC part; the 1-electron and Coulomb parts are computed
in full double precision on the CPU (via density fitting in `int3lu`).

---

## 7. Key Variables Reference

### Global State (`garcha_mod.f`)

| Variable | Type | Description |
|---|---|---|
| `Pmat_vec(MM)` | real*8, packed | Current density matrix |
| `Fmat_vec(MM)` | real*8, packed | Alpha Fock matrix (output from int3lu + g2g) |
| `Fmat_vec2(MM)` | real*8, packed | Beta Fock (open-shell) |
| `Hmat_vec(MM)` | real*8, packed | Core Hamiltonian (1-electron) |
| `Gmat_vec(MM)` | real*8, packed | G matrix (Coulomb fitting) |
| `rhoalpha(MM)` | real*8, packed | Alpha density (open-shell) |
| `rhobeta(MM)` | real*8, packed | Beta density (open-shell) |
| `told` | real*8 | Density convergence threshold (typically 1e-6) |
| `Etold` | real*8 | Energy convergence threshold |
| `ndiis` | integer | DIIS history size (typically 30) |
| `DAMP` | real*8 | Damping factor |
| `MM` | integer | M*(M+1)/2, packed matrix size |

### DIIS State (`converger_data.f90`)

| Variable | Type | Description |
|---|---|---|
| `fockm(M,M,ndiis,nspin)` | real*8 | Fock history buffer (ON basis) |
| `FP_PFm(M,M,ndiis,nspin)` | real*8 | Error vector history [F',P'] |
| `EMAT2(ndiis,ndiis,nspin)` | real*8 | Cached error matrix entries |
| `bcoef(ndiis+1,nspin)` | real*8 | DIIS coefficients + Lagrange multiplier |
| `head_idx(2)` | integer | Circular buffer head per spin |
| `fock_damped(M,M,nspin)` | real*8 | Previous damped Fock for next iteration |
| `damping_factor` | real*8 | α in `(F_new + α·F_old)/(1+α)` |
| `conver_criter` | integer | 1=damping, 2=DIIS, 3=hybrid |

---

## 8. Safe vs Unsafe GPU Kernel Changes (Summary)

### Safe (preserves DIIS trajectory)
- Warp shuffle reductions matching original volatile FP order
- Shared memory layout changes (same arithmetic)
- Loop unrolling without reordering operations
- Structural optimizations (persistent streams, memory pooling)
- Converger-side improvements (P1/P2/P3: persistent arrays, circular buffer)

### Unsafe (changes float32 bit patterns → disrupts DIIS)
- Kahan compensated summation in any kernel feeding DIIS
- `__launch_bounds__` (changes register allocation → different FP scheduling)
- Double-precision accumulators inside GPU kernels
- Any reordering of float32 additions in reduction trees
- Mixed-precision schemes that change intermediate float32 values

### Applied Fix: DGELSS replaces DGELS (commit pending)

The DIIS linear system solver was changed from DGELS (QR factorization, no
rank detection) to DGELSS (SVD-based least-squares with rank detection).

**Measured impact on ill-conditioned test case** (nearly-parallel error vectors):
- DGELS: max(|bcoef|) = **3.9 million** (amplifies noise by 4M×)
- DGELSS: max(|bcoef|) = **1.0** (minimum-norm solution, bounded)

DGELSS detects near-singular directions via SVD and truncates them, producing
minimum-norm solutions that naturally limit coefficient magnitudes. A safety
net checks max(|bcoef|) > 10,000 and falls back to using only the current
Fock (no extrapolation) if coefficients are still too large.

The fix makes DIIS robust to float32 noise pattern changes, which should
allow GPU-side precision improvements (Kahan summation, `__launch_bounds__`)
without convergence regressions. Verification with Kahan is the next step.

### The Nuclear Fix
Full double precision on GPU (`precision=1` build, `FULL_DOUBLE` macro).
This eliminates the float32 noise floor entirely, making DIIS robust to
any FP ordering changes. Half-measures create precision mismatches worse
than consistent float32.
