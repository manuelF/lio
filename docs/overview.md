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
│     if (kk > jj) del = del * sqrt(2)  [*]            │
│     good = good + del²                                │
│   enddo ; enddo                                       │
│   good = sqrt(good) / M                               │
│                                                       │
│ [*] sqrt(2) only for off-diagonal: symmetric storage  │
│     stores upper triangle only; off-diag elements     │
│     represent 2 matrix entries, diagonals represent 1 │
│                                                       │
│ Loop continues while good ≥ told OR Egood ≥ Etold     │
│ (told typically 1e-6, Etold effectively disabled=1.0) │
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
| DIIS: solve EMAT·c=b | **double** (DGELSS/SVD) | `converger_subs.f90` |
| Fock diagonalization | **double** (DSYEVD) | `typedef_operator/Diagon_datamat` |
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

This is solved via LAPACK `DGELSS` (SVD-based least-squares with rank detection).
The constraint `Σ c_k = 1` is enforced by the Lagrange multiplier row.

**Why DGELSS, not DGELS or DGESV**: Near convergence, DIIS error vectors become
nearly parallel, making EMAT nearly singular. DGELS (QR) doesn't detect rank
deficiency and produces wild coefficients (max |c_k| = 3.9 million on test case).
DGELSS truncates near-zero singular values → minimum-norm solution → bounded
coefficients (max |c_k| ≈ 1.0). A safety fallback rejects coefficients with
max |c_k| > 10,000 and uses only the current Fock matrix.

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

## 5. Convergence Tuning Experiments & Verified Components

### 5.1 DIIS Solver: DGELS → DGELSS (RESOLVED — Implemented)

**Problem**: DGELS (QR) doesn't detect rank deficiency. Near convergence, error
vectors become nearly parallel → EMAT nearly singular → coefficients blow up
(max |c_k| = 3.9M) → amplifies float32 noise by millions.

**Fix**: Replaced with DGELSS (SVD). Truncates near-zero singular values →
minimum-norm solution → bounded coefficients (max |c_k| ≈ 1.0). Added safety
fallback: if max |c_k| > 10,000, use current Fock only.

**Result**: All E2E tests pass, fosfatoQMMM 25 iters, energy unchanged. DIIS
is now robust to float32 noise pattern changes.

### 5.2 GOLD (Damping Factor) Reduction — FAILED

Reducing GOLD makes convergence **worse** on fosfatoQMMM (M=86):

| GOLD | Mixing fraction (new Fock) | SCF Iterations |
|------|---------------------------|----------------|
| 10 (default) | 9.1% | **25** |
| 8 | 11.1% | 28 |
| 5 | 16.7% | 26 |
| 2 | 33.3% | 29 |

**Root cause**: Heavy damping (GOLD=10) smooths out float32 GPU noise in the
first 2 iterations, giving DIIS cleaner starting vectors. More aggressive mixing
amplifies noise → DIIS starts with noisier vectors → needs more iterations.

**Conclusion**: GOLD=10 is optimal for float32 GPU precision. Standard DFT codes
use GOLD=1-2 because they work in full double precision. Do NOT reduce GOLD.

### 5.3 sqrt(2) Diagonal Bug in Convergence Metric (FIXED)

**Location**: `SCF.f90:722-731`

**Bug**: `sq2` factor was applied to ALL density matrix elements including diagonal.
Off-diagonals need sqrt(2) because symmetric storage stores only the upper triangle
(each off-diag element represents 2 matrix entries). Diagonals appear once and should
have factor 1.

**Fix**: `if (kk.gt.jj) del=del*sq2` (was: `del=del*sq2`)

**Impact**: Metric ~25% smaller with fix. Iteration count unchanged (25) because
the metric change doesn't affect the SCF trajectory — only when convergence is
declared. The fix is mathematically correct.

### 5.4 Verified-Correct Components (Deep Audit)

The following were audited and confirmed correct:

| Component | Location | Verified |
|---|---|---|
| DGEMM calls (transpose flags, LDA) | `converger_subs.f90`, `typedef_operator/` | Correct |
| Commutator [F',P'] = F'P' - P'F' | `commutator_gemm.f` | Correct sign & order |
| Base change X^T·A·X | `BChange_AOtoON` | 2 DGEMMs: 'T','N' then 'N','N' |
| Eigenvalue solver DSYEVD | `matrix_diagon_dsyevd.f90` | Ascending eigenvalues |
| Density build P = 2·C·C^T | `Dens_build` | Occupation 2.0 (closed), 1.0 (open) |
| Fock reset each iteration | `subm_int3lu.f90:110-112` | `Fmat = Hmat` then adds Coulomb |
| `spunpack_rho` UPLO='L' | `packed_storage.f90` | Correct (Fortran DO loop semantics) |
| `messup_densmat` / `fix_densmat` | `SCF_aux.f90` | Correct 2× off-diag scaling |
| `standard_coefs` (MO phase) | `SCF_aux.f90` | Doesn't interact with DIIS |
| DIIS stores raw (undamped) Fock | `converger_subs.f90` | Correct — fockm gets raw F' |
| EMAT symmetry & caching | `converger_data.f90` | Correct circular buffer |

### 5.5 Fock Matrix Assembly (Verified)

`Fmat_vec` is NOT explicitly zeroed. Instead, `int3lu` overwrites it:
`Fmat_vec = Hmat_vec + Coulomb_terms`, then `g2g_solve_groups` ADDS XC.
This is correct: `int3lu` sets `Fmat(k) = Hmat(k)` (line 110-112 of
`subm_int3lu.f90`), then adds Coulomb terms. g2g subsequently adds XC.

### 5.6 Not Yet Implemented — Potential Improvements

**Level Shifting**: Adds constant σ to virtual orbital energies in the ON basis:
`F'(a,a) += σ` for `a > NCO`. Increases HOMO-LUMO gap, prevents charge sloshing.
Standard technique (Saunders & Hillier 1973), used in Q-Chem, ORCA, Gaussian.
Implementation point: `SCF.f90` after `Diagon_datamat`. Typical σ = 0.1–0.5 Ha.
Default 0.0 for backward compatibility.

**EDIIS/ADIIS**: Energy-based DIIS for early iterations where standard DIIS
(commutator-based) performs poorly. ADIIS blends EDIIS (early) with CDIIS
(near convergence). Would require storing total energies per iteration.

**Anderson/Broyden Mixing**: Alternative to DIIS that uses density differences
rather than Fock matrices. May be more stable for difficult systems.

**Dynamic Damping**: Adaptive GOLD based on energy trajectory — relax when
energy decreases, tighten when it oscillates. Only applies during damping phase.

### 5.7 Key Insight: Float32 Noise Dominates Convergence Tuning

Any parameter change that would help in double precision may HURT in float32:
- Reducing damping → amplifies noise → worse DIIS start
- Kahan summation → changes noise pattern → different SCF trajectory
- Any GPU kernel change → different FP rounding → different iteration count

The float32 noise floor (~2e-6 in density RMS) is only 3× below the convergence
threshold `told=1e-6`. The last 3-5 iterations fight noise, not physics. The
DGELSS solver handles this by detecting rank deficiency in the DIIS error matrix.

ndiis=30 (vs typical 6-12) is needed because with the float32 noise floor, DIIS
must search a larger subspace to find a combination that "accidentally" produces
`rho_diff < told`.

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

### Applied Fix: DGELSS replaces DGELS (committed)

The DIIS linear system solver was changed from DGELS (QR factorization, no
rank detection) to DGELSS (SVD-based least-squares with rank detection).
See Section 5.1 for details.

### Applied Fix: sqrt(2) diagonal convergence metric (implemented)

The convergence metric in `SCF.f90:722-731` was corrected to apply the sqrt(2)
scaling factor only to off-diagonal elements. See Section 5.3 for details.

### The Nuclear Option
Full double precision on GPU (`precision=1` build, `FULL_DOUBLE` macro).
This eliminates the float32 noise floor entirely, making DIIS robust to
any FP ordering changes. Half-measures create precision mismatches worse
than consistent float32.
