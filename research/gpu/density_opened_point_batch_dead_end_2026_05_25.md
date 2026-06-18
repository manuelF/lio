# Density-opened kernel optimization attempts — Profile Results (2026-05-25)

## What I learned via nsys + ncu

`gpu_compute_density_opened` is 84.9% of CUDA time on heme open-shell
(5.65 s of 6.65 s GPU total), with 8176 instances per SCF and **huge
duration variance**: min 22 µs, **avg 690 µs, max 8.6 ms**, std-dev
1023 µs.

Two regimes:
- **Small launches (~85-400 blocks, Waves/SM = 0.35)**: achieved
  occupancy 17%, SM throughput ~30%, L1TEX-stall ~37%.
- **Big launches (~8000+ blocks, Waves/SM = 8.9)**: achieved occupancy
  **48% (≈ the theoretical max of 50%)**, **SM throughput 72%**,
  L1/TEX throughput 73%, DRAM 15%. Compute+memory balanced.

The "5% GPU utilization" observation averages over many tiny launches
where the GPU is launch-overhead-bound; the *time* is dominated by the
big launches, which are already well-tuned.

## What I tried (both dead ends)

### Point batching (POINTS_PER_BLOCK=4)
Added `gpu_compute_density_opened_batched` template with
block=(64, 4, 1). Per-point FP order preserved → bit-exact convergence.
**Result**: kernel duration rose 36 µs → 40 µs (+10%). Same achieved
warps/SM as unbatched (24 warps either way — register pressure is the
binding constraint, not block count). L1TEX stall dropped 37 % → 35 %
but the per-block reduction overhead ate the savings.

### __launch_bounds__(64, 16)
Forces ptxas to spill regs to fit 16 blocks/SM. Regs dropped 80 → 64.
**Result**: kernel duration unchanged, heme wall within noise. The
register spills cancel the occupancy gain on big launches; on small
launches the extra occupancy doesn't help because the workload itself
is too small to fill the SM.

## Why the obvious levers don't work

**Total warps/SM is invariant under reshuffles** that keep register
count constant. The kernel is reg-pressure-bound (80 regs/thread → 12
blocks/SM × 2 warps = 24 warps/SM, vs 48-warp hardware max). Any
per-block rearrangement (more warps/block, point batching) shifts the
*shape* of warp packing but not the *count*. To meaningfully raise
occupancy you have to drop register pressure substantially (target
< 32 regs, currently 80) — and the kernel needs those accumulators for
both alpha and beta paths.

The big launches are already running at 72% SM throughput. The
theoretical ceiling for *this kernel shape* is maybe 85-90%. That's
~15-20% headroom — not 2x.

## Where the real headroom lives

1. **Float2 alpha+beta texture fusion** (~10-20% kernel speedup).
   The inner loop does 2 separate `tex2D<float>` calls per j iteration
   (P_a and P_b). Storing both in one float2-typed texture and reading
   via `tex2D<float2>` halves the L1TEX request count. Memory throughput
   would drop from 72% to ~40%, compute can fill the gap. Touches:
   gather/scatter kernels, texture binding in `iteration.cu`,
   density kernel fetch macros. ~150-200 LOC change.

2. **Persistent / fused multi-group launches**. Of the 8176 launches,
   many are small (< 100 blocks each). Batching them into a single
   kernel that consumes from a work queue eliminates launch overhead
   and lifts the small-launch GPU utilization from 5% → 30+%. Risk:
   significant rewrite, per-group state in constant/global memory.

3. **Algorithmic: density-as-GEMM** (listed in MEMORY.md as candidate
   #7). Reformulating the kernel as P · F gives access to cuBLAS
   batched GEMM. Bigger restructure.

## Reproducer
```bash
cd test/LIO_test/13_Heme
LIO_OVERLAP_INT3LU_G2G=1 ncu --kernel-name regex:"gpu_compute_density_opened" \
  --launch-skip 5000 --launch-count 5 --section SpeedOfLight \
  --section Occupancy ../../../liosolo/liosolo -i heme.in -c heme.xyz -b DZVP
```

Look for the (4268, 2, 1)×(64, 1, 1) launches — those are the typical
big ones (Achieved Occupancy ≈ 48%, SM Throughput ≈ 72%).
