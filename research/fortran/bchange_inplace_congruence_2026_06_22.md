# BChange in-place congruence — kill the redundant M×M copies (multiZn pole)

**Date:** 2026-06-22
**Status:** DONE (bit-exact, shipped)
**Case:** `12_Zn_timers` (multiZn, open-shell GGA, M=2600, nmax=2, `free_global_memory=-1`,
`LIO_OVERLAP_INT3LU_G2G=1`). HW: Ryzen 7 5800X3D + RTX 3080 Ti.

## Context — where the multiZn pole actually is

multiZn is **CPU-BLAS bound**; the GPU XC kernel is hidden (total GPU ~265 ms / 21.5 s wall,
~1.2% — see [[density_opened_two_regime_2026_06_22]]). Per-iteration the exposed sequential
poles (GPU idle the whole time) are:

| Phase | time (2 iters) | what |
|---|---|---|
| Fock diagonalization | 3.09 s | 2× DSYEVD(M=2600) per iter (α,β) |
| **Conv setup — BChange AOtoON** | **2.89 s** | up to 8× M³ DGEMM/iter (dens α/β + fock α/β congruences) |
| DIIS commut | 0.92 s | already optimal post-c8f3d33a (1 DGEMM + antisym) |

## What was measured (before touching anything)

- **Thread scaling is flat**: OPENBLAS_NUM_THREADS 8 vs 16 → BChange 3.01 vs 3.04 s, diag
  3.34 vs 3.23 s. Core-saturated, not a config problem.
- **CPU FP64 DGEMM is at peak**: a microbench of one congruence (2 GEMM, M=2600) = **0.191 s
  @ 369 GFLOP/s** — essentially the 5800X3D's FP64 ceiling. The DGEMM math is floored.
- **GPU FP64 offload is marginal/negative**: 3080 Ti FP64 peak ≈ 530 GFLOP/s (1:64 consumer
  rate) vs CPU 369 → ~1.3× on the GEMM, eaten by PCIe transfer of 54 MB matrices + the fact
  the result feeds DSYEVD immediately (no overlap slack). Not worth it on a consumer GPU.
  (Consistent with the cuSOLVER-DSYEVD note: 2.4× *slower* at M=364.)
- **But the wrapper, not the math, was the slack**: replicating the old call path (fresh
  `Mato` allocation + 3 full M×M copies) measured **0.263 s** vs 0.191 s pure DGEMM →
  **~25–30 % of BChange was data movement, not flops.**

## The redundancy (old path, per congruence)

`BChange_AOtoON_r` → `change_base_rr` → `basechange_d_gemm`:
1. `allocate(Dmat)` + `Dmat = data_AO`         (alloc + copy 1)
2. `Mato = basechange_gemm(...)`               (allocates Mato, 2 DGEMM)
3. `Dmat = Mato`                               (copy 2, Mato auto-freed)
4. `Sets_data_ON(Dmat)` → `data_ON = Dmat`     (copy 3)

= 2 allocations + 3 M×M copies wrapped around the 2 DGEMMs.

## Fix (bit-exact)

The second DGEMM's destination is **not one of its operands** (`C = Matm·X`, inputs Matm and
X), so the congruence runs fully **in place** on the result buffer with only the persistent
`Matm` scratch.

1. New `basechange_d_gemm_inplace(M, Mat, Umat, mode)` in `mathsubs/basechange_gemm.f90` —
   identical two DGEMMs, same order, writes back into `Mat`. No `Mato` alloc, no copy-back.
2. `cumat_bchange.f90::change_base_rr` (non-CUBLAS path) calls it instead of
   `input_matrix = basechange_gemm(...)` — drops 1 alloc + 1 copy at **every** `_r` call site
   (SCF, TD fock, transport).
3. `BChange_AOtoON_r` / `BChange_ONtoAO_r` seed `data_ON`/`data_AO` directly via the existing
   setter and transform it in place — drops the separate `Dmat` scratch (1 alloc + 1 copy).

Net: 2 allocs + 3 copies → 0 extra allocs + 1 unavoidable seed copy.

Numerically identical: same DGEMM calls on the same data in the same order. The complex (`_x`)
TD paths were left untouched (not the multiZn pole; separate optimization history).

## Result

- **Conv setup — BChange AOtoON: 2.89 → 2.47 s (−15 %)**; SCF-acceleration-setup 3.95 → 3.57 s;
  Iteration 11.1 → 10.77 s.
- **Bit-exact**, all deterministic e2e PASS: `00_agua` (closed), `02_Fe3H2O6` (open, restart —
  Energy/Forces/Mulliken/Dipole), `03_fosfatoQMMM`, `05_TDDFTField`, `07_TDDFTHCL`. (multiZn
  itself is not a valid oracle — `free_global_memory=-1` rebalancer is run-to-run nondeterministic.)

## What is now floored

Diag (DSYEVD, basis-locked, NCO/M≈0.58 so subset-DSYEVR loses + perturbs convergence), the
DGEMM math (CPU at FP64 peak, GPU offload marginal on consumer FP64), and DIIS commut (already
1-GEMM+antisym). Remaining BChange has 1 seed copy left (~14 ms/iter, ~0.6 %) — not worth the
invasive separate-src/dst `change_base` refactor. **The CPU-BLAS pole is now genuinely at the
algorithmic/BLAS floor.**
