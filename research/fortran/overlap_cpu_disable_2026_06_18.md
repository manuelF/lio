# int3lu/g2g overlap is counterproductive on CPU-only builds — DONE 2026-06-18

**Status:** DONE. Shipped. Bit-identical energy/forces/dipole/mulliken.

## TL;DR

`LIO_OVERLAP_INT3LU_G2G=1` runs `int3lu` (Coulomb fit + Fock, CPU BLAS) in one
OpenMP section concurrently with `g2g_solve_groups` (XC Fock) in another. This is
a **GPU optimization**: it hides int3lu's cheap CPU BLAS work *under the GPU XC
solve*. On a **CPU-only build (`make cuda=0 cpu=1`) the XC solve runs on the CPU**,
where it is core-bound and ~10× larger than int3lu, so splitting the cores between
the two sections starves XC and the overlap is a **net loss** versus running both
phases sequentially with all physical cores each.

Fix: gate the overlap activation on `g2g_gpu_threads() > 0`. On CPU builds it now
falls back to the (faster) sequential path and prints a one-line notice.

## Measurement (fosfatoQMMM, 27 iters, Ryzen 7 5800X3D, 8 physical cores)

Per-iteration overlap log (`verbose>3`) on the pre-fix CPU binary:

```
[overlap] int3lu= 21.2ms  g2g= 161.0ms  idle= 139.8ms
```

The int3lu section finishes in 21 ms then idles 140 ms while the core-bound XC
section grinds on with fewer cores. The two are wildly imbalanced — overlapping
buys nothing and costs XC its cores.

| Build / mode | Fock-build phase ("Fock integrals", 27 iters) |
|---|---|
| overlap ON (pre-fix) | **4.49 s** (merged "Coulomb fit + Fock") |
| sequential (post-fix, env=1) | **~3.8 s** (Coulomb 0.36 s + XC 3.48 s) |

≈ **15 % off the Fock-build phase**, which is ~88 % of the SCF iteration cost.
XC alone is 3.48 s = 129 ms/iter and is THE per-iteration pole on CPU.

## Why the overlap can never win on CPU (the math)

Sequential: `int3lu(13 ms, all cores) + XC(129 ms, all cores) = 142 ms/iter`.
Overlap: `max(int3lu_capped, XC_starved) ≈ 161 ms/iter`.

XC throughput scales with cores; int3lu is only 13 ms at full cores. Handing any
cores to int3lu costs XC more than the ≤13 ms it could save, so the optimum on
CPU is always sequential with all cores per phase. On GPU the trade is reversed
(int3lu's CPU cores are free while the GPU does XC), so the overlap stays enabled
when `gpu_threads > 0` (cuda=1 builds) — no behavior change there.

## Implementation

- `g2g/init.cpp`: new `extern "C" int g2g_gpu_threads_()` returning
  `G2G::gpu_threads` (0 on CPU-only / no-CUDA-device builds). Also gated the
  init-time `[overlap] auto OMP=.. BLAS=..` notice on `gpu_threads > 0`.
- `lioamber/gpu_interface.f90`: `integer function g2g_gpu_threads()` interface.
- `lioamber/SCF.f90`: overlap activation now requires
  `... .and. g2g_gpu_threads() > 0`; added an `else if` notice when the overlap
  is requested but skipped for lack of a GPU.

## Validation

`03_fosfatoQMMM` Energy/Forces/Mulliken/Dipole all OK; `00_agua` and open-shell
`02_Fe3H2O6` (restart) all PASS. Final energy unchanged to all printed digits
(`-2148.6508457`).

## See also

- [[overlap_autotune_2026_05_03]] — the auto-tuned thread split this supersedes on CPU.
- `../cpu/int3lu_gpu_offload_2026_06_10.md` — the GPU-resident int3lu path (where overlap still helps).
