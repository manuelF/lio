# Optimization: Kernel Batching (Multi-Group Kernel)

## Summary
The system partitions grid points into `PointGroup` objects processed sequentially.
For small groups, each kernel launch uses only a fraction of the available SMs.

**GTX 1080 context**: 20 SMs. A group with P=64 points launches
`gpu_compute_density` with a grid of 64 blocks (one per point) — using all 20 SMs
but with only 3–4 waves. A group with P=16 uses only 16/20 = 80% of SMs.

The batched kernel approach processes multiple groups in a single launch, merging
their work into a larger, more efficient grid.

## Assessment Correction

The original file cited "A100 has 108 SMs" as motivation. **For our GTX 1080 (20 SMs),
the under-utilization problem is less severe** — a group with P=32 still launches
32 blocks ≥ 20 SMs. The real benefit of batching for SM 6.1 is:
1. Reduced kernel launch overhead (5–10 µs × N_groups CPU time saved).
2. Better occupancy for very small groups (P<20).
3. Reduced scheduler overhead.

**Priority**: Lower than on A100. Stream sharding (2–4 streams) achieves similar
occupancy improvement with less implementation complexity. Consider batching only
after stream sharding is validated.

## Proposal — Block-Per-Group Assignment (Simplest Variant)

Each group gets one thread block in the batched kernel. Threads within the block
process points. This avoids inter-group synchronization:

```cpp
// New kernel: batched_compute_density
__global__ void batched_compute_density(
    float** function_values_ptrs,   // Array of per-group pointers
    float** rmm_ptrs,               // Array of per-group RMM pointers
    float** density_out_ptrs,       // Array of per-group density output
    int*    group_m_arr,            // M per group
    int*    npoints_arr) {          // P per group
  int group_id = blockIdx.x;
  int point    = threadIdx.x;
  int M = group_m_arr[group_id];
  int P = npoints_arr[group_id];
  if (point >= P) return;
  const float* F   = function_values_ptrs[group_id];
  const float* rmm = rmm_ptrs[group_id];
  // ... compute density for this point ...
}

// Launch: one block per group
batched_compute_density<<<N_groups, MAX_POINTS_PER_GROUP>>>(ptrs...);
```

**Limitation**: all blocks use `MAX_POINTS_PER_GROUP` threads, wasting threads
for groups with P < MAX_POINTS_PER_GROUP. Use `cudaLaunchCooperativeKernelMultiDevice`
or variable-block-size launches if heterogeneity is high.

## Memory Layout Requirement

All groups must have their data on GPU before the batched launch. Currently, data
is transferred and processed group-by-group. For batching to work:
1. Pre-transfer ALL group function values to GPU before any density computation.
2. Store per-group pointers in a GPU-side pointer array.
3. This requires `N_groups × max_group_size × M × 4 bytes` of VRAM simultaneously.
   For 100 groups with M=30, P=512: 100 × 512 × 30 × 4 = 614 MB. Feasible on 8 GB.

This is a significant memory commitment and may limit system size.

## Persistent Thread Alternative (for functions kernel)

For `gpu_compute_functions` (most expensive for large systems), a persistent kernel
consumes groups from a queue:
```cpp
__global__ void persistent_compute_functions(GroupQueue* queue, ...) {
  while (true) {
    int gid = atomicAdd(&queue->head, 1);
    if (gid >= queue->total) break;
    // Process group gid...
  }
}
// Launch: exactly 20 blocks (one per SM)
persistent_compute_functions<<<20, FUNCTIONS_BLOCK_SIZE>>>(queue, ...);
```
This guarantees 100% SM utilization at all times with zero launch overhead.
Downside: requires all group data in GPU RAM simultaneously.

## Impact (Corrected for GTX 1080)

| Scenario | Estimated Speedup |
|---|---|
| Many groups with P<20 (tiny cubes) | 1.5–3× launch overhead saving |
| Typical mixed groups P=64–512 | 1.1–1.3× |
| Few large groups P>512 | <1.05× (already well-utilized) |

The original "2x–5x" estimate applied to A100 where SM underutilization is severe.
For GTX 1080 with 20 SMs: **1.2–2× for small-group-dominated systems**.

## Difficulty Assessment
**High** (accurate) — Very high complexity due to:
1. Memory layout restructuring (all groups in GPU RAM simultaneously).
2. Per-group pointer arrays on GPU (indirect access).
3. Variable group sizes → variable block sizes (hard to express in CUDA).
4. `GlobalMemoryPool` must reserve memory for all groups upfront.

**Recommendation**: Implement stream sharding (4 streams) first — achieves 60–80%
of the benefit at 20% of the complexity. Batching is worth pursuing only if profiling
shows stream sharding leaves >20% SM idle time.

## Files to Modify
- `g2g/cuda/iteration.cu`: Major restructuring of dispatch loop.
- `g2g/cuda/kernels/functions.h`: Add group-ID indirection layer.
- `g2g/partition.h`: Flattened GPU buffer for all group data.
- `g2g/global_memory_pool.h`: Reserve-all-upfront mode.

## Estimations
- Small-group-dominated: **1.5–2× speedup** (corrected from 2–5×).
- Typical LIO workloads: **1.1–1.3×**.
- Implementation: 2–3 weeks (major architecture change, high risk).
- Ratio of benefit to effort: **Low**. Do stream sharding first.
