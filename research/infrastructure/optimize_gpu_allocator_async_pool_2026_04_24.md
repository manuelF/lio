---
status: DONE
date: 2026-04-24
impact: 5% wall (fosfatoQMMM)
---

# GPU Allocator: stream-ordered async pool (Tier 1)

## Result

Replaced `cudaMalloc`/`cudaFree` in `g2g/matrix.cpp` (CudaMatrix resize, deallocate,
and three `operator=` overloads — 14 call sites total) with `cudaMallocAsync`/
`cudaFreeAsync` on stream 0 (legacy default stream). Set the device's default
memory pool release threshold to `UINT64_MAX` in `g2g_init_` so freed blocks stay
cached instead of being trimmed back to the driver after every `cudaFreeAsync`.

This is **Tier 1** of the allocator plan in `optimize_gpu_allocator.md`: rely on
the CUDA driver's stream-ordered memory pool (CUDA 11.2+) instead of writing a
custom arena allocator. Zero lifetime-classification work, zero custom allocator
debugging surface.

## Measured impact (fosfatoQMMM, RTX 3080 Ti, 25 SCF iters)

Wall time, 10-run median, hot disk cache:

| Variant | Median | Range |
|---|---|---|
| Baseline (`cudaMalloc`/`Free`) | **2.19 s** | 2.15 – 2.28 |
| Async pool (`cudaMallocAsync`/`FreeAsync`) | **2.09 s** | 2.06 – 2.10 |
| **Saved** | **~100 ms (5%)** | |

CUDA API breakdown (nsys, single run):

| API | Baseline calls / time | Pool calls / time | Δ time |
|---|---|---|---|
| `cudaMalloc` | 1298 / 31.6 ms | 0 | −31.6 ms |
| `cudaFree` | 1298 / 69.1 ms | 0 | −69.1 ms |
| `cudaMallocAsync` | 0 | 1298 / 13.5 ms | +13.5 ms |
| `cudaFreeAsync` | 0 | 1298 / 1.4 ms | +1.4 ms |
| **Total alloc/free** | **100.7 ms** | **14.9 ms** | **−85.8 ms** |

Per-call cost: `cudaFree` median 3.4 µs → `cudaFreeAsync` 0.94 µs (3.6× faster).
`cudaMalloc` median 3.9 µs → `cudaMallocAsync` 1.4 µs (2.8× faster). One-time
pool warmup absorbed in the first allocation (5.7 ms outlier).

GPU kernel time **identical** to baseline (cudaStreamSync 418→419 ms,
gpu_compute_density 259.7→260.3 ms). Pool change is a pure CPU-side
allocator-overhead reduction, no kernel side-effects.

## Correctness

All 30 tests pass (22 unit + 8 e2e). fosfatoQMMM dipole/Mulliken/forces/output
match references.

## Files changed

- `g2g/matrix.cpp` — 14 raw API call sites → async variants.
- `g2g/init.cpp` — set `cudaMemPoolAttrReleaseThreshold = UINT64_MAX` per device.

## What's left

- `cudaMallocArray` / `cudaFreeArray` (texture arrays at `iteration.cu:278,766-767`)
  — 45 calls, ~1 ms total. No async variant exists; would need per-PointGroupGPU
  caching like function values. Not worth pursuing alone.
- The remaining 14.9 ms is small enough that Tier 2 (hoist AINT temporaries to
  static) and Tier 3 (custom arena) are no longer worth the implementation cost.
  Tier 1 closed this lever.

## Why this works

The CUDA driver's default memory pool is a real pool: `cudaFreeAsync` returns the
block to the pool, `cudaMallocAsync` reuses pooled blocks without going to the
driver. The default release threshold is **0**, meaning blocks are trimmed back
to the driver immediately on free — which is why the threshold must be raised
explicitly. With `UINT64_MAX`, the pool grows as needed and never shrinks during
the run.

Stream ordering: all CudaMatrix allocations and the kernels that read/write them
land on stream 0 (legacy default), which serializes implicitly with persistent
streams (`transpose_stream_1/2`). No additional synchronization needed.

## Related

- Plan doc: `optimize_gpu_allocator.md` — Tier 1 of the three-tier proposal.
- `g2g/CLAUDE.md` "GPU memory caching (`fgm=-1`)" — orthogonal: that one
  eliminates per-iter allocations of function-value buffers; this one cuts the
  CPU cost of the allocations that remain (first-iter setup + AINT post-SCF).
