# Optimization: Density Kernel Texture Reads — INVESTIGATED, tex2D WINS

## Status: CLOSED — tex2D must be kept (confirmed on both Pascal and Ampere)

**Date investigated:** 2026-03-20 (Pascal SM 6.1), **2026-04-07 (Ampere SM 8.6)**
**Conclusion:** `__ldg` is slower than `tex2D` on both Pascal (36% regression) and
Ampere (2.5× regression). The root causes differ by architecture but the conclusion
is the same: keep tex2D for density kernel RMM reads.

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

---

## Ampere SM 8.6 Results (RTX 3080 Ti, 2026-04-07)

Re-tested on Ampere, which has a unified L1/texture cache (128 KB). The hypothesis was
that with a unified cache, the 2D tiling advantage might disappear and `__ldg` could win
by eliminating CUDA array allocation + `cudaMemcpy2DToArray` overhead.

### Implementation

Same approach as Pascal test: replaced `tex2D` with `__ldg(&rmm_ptr[row * stride + col])`
in energy.h, energy_open.h, energy_derivs.h. Removed texture infrastructure from
iteration.cu and partition.h/cpp. Used `fetch_rmm` macro for clean replacement.

### ncu metrics for `gpu_compute_density` (largest GPU group, GGA)

| Metric | tex2D | `__ldg` | Delta |
|--------|-------|---------|-------|
| **L1 hit rate** | **83.68%** | **93.17%** | **+9.5 pp (better)** |
| **Compute throughput** | **94.25%** | **59.79%** | **−34.5 pp** |
| Memory throughput | 94.25% | 87.13% | −7.1 pp |
| Occupancy (achieved) | 64.67% | 59.21% | −5.5 pp |
| Registers | 48 | 48 | same |
| **Time per call** | **~233 µs** | **~584 µs** | **+151% (2.5× slower)** |

### Why `__ldg` is slower despite better L1 hit rate (Ampere)

On Ampere, the root cause is **different from Pascal**. L1 hit rate actually *improved*
by 9.5 pp. The regression comes from **software address computation overhead** in the
tight inner loop.

With `tex2D`, the texture unit performs 2D→linear address translation in hardware. The
inner loop in `gpu_compute_density` (unrolled 4×, DENSITY_BLOCK_SIZE iterations) does
~14 FP ops per density matrix fetch:

```
rdm = tex2D(tex, col, row)     // hardware address translation, ~0 ALU cost
w += rdm * fj_val              // 1 FMA
w3  += fgj_sh[j]  * rdm        // 3 FMA (vec3)
ww1 += fh1j_sh[j] * rdm        // 3 FMA
ww2 += fh2j_sh[j] * rdm        // 3 FMA
// + 4 more for second thread pair (rdm2)
```

With `__ldg`, each fetch adds a multiply-add for address computation:

```
addr = row * stride + col       // 1 IMUL + 1 IADD (software)
rdm = __ldg(&rmm_ptr[addr])    // load through L1
```

Two extra integer ops per fetch in a 14-FP-op loop body represents ~14% instruction
pressure increase. Since the kernel was already at 94% compute throughput, the extra
instructions push it into instruction throughput bottleneck territory, dropping compute
utilization to 60%.

### Exception: `gpu_compute_density_derivs` IMPROVED with `__ldg`

| Metric | tex2D | `__ldg` | Delta |
|--------|-------|---------|-------|
| Time per call | 1.79 ms | 0.80 ms | **−55% (2.2× faster)** |

This kernel has a fundamentally different access pattern: cooperative loads into shared
memory with `__syncthreads()` between load and compute phases. The address computation
is done during the load phase (low compute pressure), and the compute phase only reads
from shared memory. However, at only 5.6% of GPU time, this improvement doesn't justify
a mixed tex2D/__ldg codebase.

### Why the causes differ by architecture

| | Pascal SM 6.1 | Ampere SM 8.6 |
|---|---|---|
| L1 hit rate | 82.85% → 76.48% (−6.4 pp) | 83.68% → 93.17% (+9.5 pp) |
| Main bottleneck | Memory latency (more misses) | Instruction throughput (address computation) |
| Regression severity | 36% | 151% (2.5×) |

On Pascal, the texture unit's 2D Morton-order tiling provides genuine cache locality
that linear __ldg cannot match. On Ampere, the unified L1 cache actually handles
linear access patterns *better* (higher hit rate), but the software address computation
overhead in the tight inner loop is the dominant cost.

---

## Conclusion: DO NOT replace tex2D with `__ldg` for density kernel RMM reads

Tested on two architectures spanning 4 GPU generations. Both show regressions, for
different architectural reasons. The tex2D approach is the right choice for this
access pattern regardless of GPU generation.

The only scenario where this might change is if the inner loop's compute intensity
increases significantly (e.g., higher angular momentum, more terms per fetch), which
would amortize the address computation cost.

## Part A (Warp Shuffle) — COMPLETED

The warp shuffle optimization from the original proposal was completed separately in
commit `ac87eef0`. It reduced shared memory from 2560→256 bytes (LDA) and achieved
100% theoretical occupancy. This was a clear win, unlike Part B (texture removal).
