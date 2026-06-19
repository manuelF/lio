# BLAS3 (SSYMM) for the CPU density matvec — DEAD END 2026-06-19

**Status:** REJECTED. Microbench-faster in isolation, net-slower in production.

## Context

After the TD merge-off win ([[td_merge_cpu_disable_2026_06_18]]), the TD-on-CPU XC
solve is dominated by `solve` itself. Instrumenting `Partition::solve`'s per-phase
timers (TD chloride, open-shell, m=21 groups) gave, over the predictor XC:

- density matvec (`cpu_compute_density_gga_batch`, both spins): **59%**
- functional eval (`calc_ggaOS`, PBE, per-point): **41%**
- `compute_functions`: 0 (cached); rmm accumulate: small

The matvec computes `W_k = RMM · T_k` for the 10 GGA channels (value + 3 grad +
6 hessian — all genuinely needed). It's a hand-tiled, point-vectorized, triangle-
exploiting reduction.

## The idea and why it looked promising

A standalone SGEMM microbench at the production shape (M=10·np=38800, N=K=21,
single-thread) ran the matvec in **0.115 ms at ~300 GFLOPS** — the large M makes
OpenBLAS efficient even at tiny K=21, far above the hand kernel. So BLAS3 SSYMM
(W = T · RMMh, non-doubled symmetric RMMh) looked like a 5-10x matvec win.

## Why it's actually slower

Implemented and measured end-to-end: **TD-Pred-XC 15.1 → 16.9 s (~12% slower)**.
The microbench only timed the matmul. The full BLAS reformulation must:
1. **materialize W** (10·np·m floats) to memory — the tiled kernel keeps W in
   registers, fused per i-tile;
2. run a **separate epilogue pass** over W (cancellation forces double accumulation).

For these small-m grids the materialization + extra pass cost more than the matmul
saves. The fused register-blocked tiled kernel is already the right structure for
small m. (BLAS3 would only pay off if the epilogue could be fused into the GEMM,
which it can't.)

## Secondary finding: float full-W loses accuracy at large m

The full-symmetric reformulation has ~2x the intermediate magnitude of the tiled
triangular sum, so in **float** the per-point sums lose accuracy as m grows: a
densbad probe showed up to **~6% relative error on the hessian at m~100** (fine at
m≤21). Any BLAS path would have to gate on small m or use double — another strike.

## Tripwire: fosfato SCF is bistable under caching

While chasing a phantom "11 mHa regression" I confirmed it was **not** caused by
the BLAS kernel or the `-lopenblas` link: clean `b0b8393d` with none of the change
also gives `-2148.6617 / 37 iters`, while earlier in the same session it gave
`-2148.6508 / 27 iters`. With `free_global_memory=-1` (caching on), the
timing-dependent `rebalance()` lets fosfato fall into one of **two SCF solutions
~11 mHa apart**. **Do not validate density/XC changes on fosfato's SCF "Final
energy" — it is bistable.** Use a deterministic case (`free_global_memory=0`) or
the TD chloride dipole test. (The densbad probe had already proven the BLAS density
matched the tiled kernel to float precision per-call.)

## Verdict

The CPU density kernel is already near-optimal for the small-m TD regime. No
non-parallel, non-reordering win there. The remaining 41% (PBE functional) is
scalar transcendentals — only approximation/vector-math would move it (out of
scope). The shipped TD lever stays the merge-off (-66%).
