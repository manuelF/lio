# int3lu is DRAM-bandwidth-bound on multiZn — CPU threading is a DEAD END (2026-06-01)

**Status:** DONE (diagnosis) / threading REJECTED with data. Real lever = GPU-resident offload (OPEN, risky).
**Case:** `test/LIO_test/12_Zn_timers` — 100 Zn, open-shell, M=1500, Md=5500, 2 SCF iters.

## What was asked
Profile *why* `int3lu` ("Coulomb fit + Fock", `lioamber/faint_cpu/subm_int3lu.f90`)
takes so long on multiZn and optimize it, keeping safety.

## Full profile (multiZn, total 25.0s, 2 iters)
One-time setup (Initialize SCF 9.6s):
- **Coulomb precalc (int3mem): 5.05s** — single biggest line, already OpenMP-parallel.
- Overlap diagonalization: 2.9s.
- Coulomb G matrix: 0.82s (G invert 0.48s).

Per-iteration (Iteration 12.7s for ~2-3 calls):
- SCF acceleration setup: 4.55s (BChange AOtoON 2.94s + DIIS commut 1.47s).
- SCF Fock diagonalization: 3.12s.
- **Fock integrals (int3lu): 1.65s** ≈ **0.54s/call**.
- SCF acceleration: 1.25s.

So in a single 2-iter run int3lu is only ~7% of wall (int3mem/diag/accel dominate),
**but** int3lu recurs every SCF iteration of every MD/TD step, so it is the right
per-iteration target for production.

## Why int3lu is slow here — MEASURED
`int3lu` MEMO path is already pure BLAS-2 (DGEMV/SGEMV/DSPMV). Per-call breakdown
(system_clock instrumentation, since the Fortran `g2g_timer` for int3lu isn't printed):

| phase | time | reads |
|---|---|---|
| gemvD = DGEMV('N')+DGEMV('T') on `cool` | 0.38s | cool (4.16 GB double) ×2 |
| gemvS = SGEMV('N')+SGEMV('T') on `cools` | 0.16s | cools (1.64 GB single) ×2 |
| dspmv (Ginv,Gmat) + ddot | 0.016s | |
| gather/scatter (random rho/Fmat index) | ~0.003s | |
| **tot** | **0.54s** | ~12 GB streamed/call |

`cool`/`cools` sizes verified: size(cool)=520,542,000 = Md·kknumd (kknumd=94644),
size(cools)=410,718,000 = Md·kknums (kknums=74676). Peak process RSS ~4.5–13 GB.

Effective bandwidth: ~12 GB streamed in 0.54s ≈ **22 GB/s** = single-core DDR4 limit
on the 5800X3D. **int3lu is memory-bandwidth-bound** streaming the two constant
integral tables, once for the Rc ('N') pass and once for the Fock ('T') pass
(the two passes are separated by the `af = Ginv·Rc` solve, so they cannot fuse).
The tables are huge here purely because of system size (100 Zn) — for small
molecules they are tens of MB and int3lu is invisible.

## DEAD END: multi-threading the GEMV (REJECTED with data)
OpenBLAS `dgemv`/`sgemv` run **single-threaded** for these shapes here
(OPENBLAS_NUM_THREADS 1 vs 16 → identical 0.374s). Hand-parallelized all four
GEMVs with `!$omp parallel` + per-thread contiguous column chunks calling
`dgemv`/`sgemv` on the chunk (the 'T'/Fock pass is bit-exact this way — each output
element is one independent column dot; the 'N'/Rc pass needs a partial-sum reduction,
not bit-exact). Swept OMP_NUM_THREADS with OPENBLAS_NUM_THREADS=1:

| threads | int3lu tot | gemvD |
|---|---|---|
| 1 | 0.530s | 0.372s |
| 2 | 0.51s  | 0.345s |
| 4 | 0.52s  | 0.36s  |
| 8 | 0.53s  | 0.36s  |

~8% at 2 threads, then flat → the memory controller saturates by ~2 cores at
~30 GB/s; extra cores do nothing. **Not a viable lever.** Reverted (also avoids
nested-OMP fragility inside the `LIO_OVERLAP_INT3LU_G2G` parallel section and the
non-bit-exact 'N' reduction in the convergence-sensitive Fock path).

## The g2g overlap cannot hide int3lu here
With `LIO_OVERLAP_INT3LU_G2G=1` the per-iter line reads:
`int3lu=540ms  g2g=124ms  idle=410ms`. The closed-shell XC `g2g_solve` is only
124ms, so int3lu is the exposed pole and the **GPU sits idle ~410ms/iter** during it.

## Real lever (OPEN, not implemented — risk + memory)
`cool`/`cools` are **constant across SCF iterations** and total 5.8 GB, which fits
in the 3080 Ti's 12 GB. Upload once after int3mem, run the 4 GEMVs as GPU-resident
cuBLAS (~900 GB/s vs ~22 GB/s) using the otherwise-idle GPU; only the Md/kknum
vectors transfer per call. Potential 0.54s → ~0.05s.
**Blockers / why not shipped:**
- VRAM: 5.8 GB cool/cools + ~5.7 GB density fgm cache ≈ 11.5 GB — borderline on
  12 GB, impossible on 8 GB GTX-1080-class. Must guard on free VRAM with CPU fallback
  (conflicts somewhat with the project "no-toggle, ship-unconditionally" rule, since
  behavior would be hardware-dependent).
- FP: cuBLAS GEMV differs from CPU BLAS at ulp; the 'T' pass feeds Fmat directly, and
  per the heme DGELS rank-deficiency findings any ulp-level Fmat perturbation can shift
  iteration counts on sensitive systems. Needs full e2e + multi-OMP iter-count check.

## Verdict
`int3lu` MEMO path is already near-optimal as a CPU BLAS-2 op and is DRAM-bandwidth
bound; there is **no safe CPU-side speedup** (threading, precision, fusion all ruled
out). The only real win is GPU-resident offload of the constant tables, which is a
larger change gated on VRAM and FP-convergence validation.
