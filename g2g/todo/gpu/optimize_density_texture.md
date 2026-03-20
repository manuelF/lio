# Optimization: Density Kernel Texture Reads — INVESTIGATED, tex2D WINS

## Status: CLOSED — tex2D must be kept on Pascal SM 6.1

**Date investigated:** 2026-03-20
**Conclusion:** `__ldg` is **36% slower** than `tex2D` for `gpu_compute_density` on GTX 1080.
The texture cache's 2D spatial locality is a genuine hardware advantage for this access pattern.

## What was tried

Replaced `tex2D<float>(rmm_tex, col, row)` with `__ldg(&rmm_ptr[col * stride + row])`
in `energy.h`, `energy_open.h`, and `energy_derivs.h`. Removed all texture infrastructure
(cudaArray, cudaTextureObject, cudaMemcpy2DToArray) from `iteration.cu` and `partition.h/cpp`.

All unit tests passed (98/98). Functional correctness was verified. SCF converged in 24-27
iterations (varies due to changed float32 bit patterns, but within acceptable range thanks
to DGELSS solver).

## Measured performance (fosfatoQMMM, 25 SCF iters)

| Metric | tex2D | `__ldg` | Delta |
|--------|-------|---------|-------|
| Wall time | 5.83s | 6.28s | **+7.7%** |
| gpu_compute_density total | 1047ms | 1438ms | **+36%** |
| gpu_compute_density avg/call | 534µs | 760µs | **+42%** |

## Profiled root cause: cache hit rate

nvprof hardware metrics for `gpu_compute_density`:

| Metric | tex2D | `__ldg` | Delta |
|--------|-------|---------|-------|
| **Unified Cache Hit Rate** | **82.85%** | **76.48%** | **-6.4 pp** |
| L2 Hit Rate (Tex Reads) | 53.80% | 60.64% | +6.8 pp |
| Global Load Efficiency | 61.08% | 74.54% | +13.5 pp |
| Achieved Occupancy | 0.508 | 0.506 | ~same |
| **stall_memory_dependency** | **31.50%** | **40.99%** | **+9.5 pp** |
| stall_exec_dependency | 20.82% | 17.92% | -2.9 pp |

## Why tex2D wins on Pascal

The RMM access pattern is `data[col * stride + row]`:
- Adjacent threads read adjacent rows within the same column → coalesced
- The inner bj-loop increments `col` each iteration → stride-separated addresses

**tex2D** maps 2D coordinates through a Morton/Z-order space-filling curve in the texture
cache. Nearby (row, col) pairs share cache lines even when linear addresses are `stride`
apart. This yields 82.85% L1 cache hit rate.

**`__ldg`** uses the same physical cache hardware but with linear addressing. Column
increments jump by `stride` floats → more cache line evictions → 76.48% hit rate. The
6.4 pp drop means ~6% more cache misses, each costing 200+ cycles. Since the kernel is
memory-latency-bound (stall_memory_dependency is the dominant stall reason), this
translates to a 36% wall-time increase.

## When `__ldg` might work

On **Volta+ (SM 7.0+)** the L1 cache is larger (128KB vs 48KB on Pascal) and has different
caching policies. The `__ldg` approach might be neutral or positive there. But on Pascal
SM 6.1, the texture cache's 2D tiling is critical for this workload.

## DO NOT re-attempt this optimization on Pascal hardware.

The 17ms saved by eliminating texture setup infrastructure is completely dwarfed by the
380ms increase in kernel execution time. The texture approach is architecturally correct
for this access pattern on this hardware generation.

## Part A (Warp Shuffle) — COMPLETED

The warp shuffle optimization from the original proposal was completed separately in
commit `ac87eef0`. It reduced shared memory from 2560→256 bytes (LDA) and achieved
100% theoretical occupancy. This was a clear win, unlike Part B (texture removal).
