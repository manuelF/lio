# Optimization: Stream Sharding for Point Groups

## Summary
The `Partition` class processes multiple `PointGroup` objects sequentially on the
default CUDA stream. For each group, the pipeline is:
`compute_functions → transpose → gpu_compute_density → accumulate → rmm_update`

All kernels serialize behind each other. Two sources of wasted time:
1. CPU stalls in `get_rmm_input` while the GPU is idle between groups.
2. Small groups (P=64, M=10) launch grids with only a few blocks, leaving most of
   the GTX 1080's 20 SMs idle.

Stream sharding processes multiple groups concurrently on separate CUDA streams,
enabling Copy Engine / Compute Engine overlap and multi-group SM utilization.

## GTX 1080 Concurrency Characteristics (SM 6.1)

- **20 SMs** (not A100's 108). Small groups with <20 blocks already underutilize.
- **1 Copy Engine** (bidirectional, can overlap with compute): H2D and D2H can
  proceed simultaneously with kernel execution on a different stream.
- **HyperQ (MPS)**: Pascal supports up to 32 concurrent hardware work queues.
  Multiple streams genuinely execute concurrently on SM 6.1.
- Memory contention: all streams share the same L2 (1.5 MB). Running 2 large groups
  concurrently doubles L2 pressure and may hurt performance. Only beneficial when
  groups are small.

**Rule of thumb**: Stream sharding helps when each group's kernels use <50% of
available SMs (i.e., grid size < 10 blocks). For LIO typical groups, this often
applies to "cube" groups but not large "sphere" groups.

## Proposal

### Phase 1 — Prerequisite: Move `get_rmm_input` to GPU

Stream sharding's CPU-GPU overlap benefit depends on the CPU having work to do
while the GPU runs a kernel. Currently, `get_rmm_input` (the biggest CPU work)
runs BEFORE the GPU kernel for that group — not concurrently with a previous group.

After `optimize_rmm_gather_gpu.md`, the gather is async on GPU, freeing the CPU to
submit the next group's kernels immediately.

### Phase 2 — Stream Pool Implementation

In `Partition`:
```cpp
// g2g/partition.h:
static constexpr int NUM_STREAMS = 4;
cudaStream_t streams[NUM_STREAMS];

// g2g/partition.cpp (constructor):
for (int i = 0; i < NUM_STREAMS; i++)
  cudaStreamCreate(&streams[i]);
```

In `Partition::solve`:
```cpp
int stream_idx = 0;
for (int ind = 0; ind < total_groups; ind++) {
  cudaStream_t s = streams[stream_idx % NUM_STREAMS];
  groups[ind]->solve(timers, ..., s);
  stream_idx++;
}
// Wait for all streams to complete
for (int i = 0; i < NUM_STREAMS; i++)
  cudaStreamSynchronize(streams[i]);
```

### Phase 3 — Pass Stream Through solve_closed

`iteration.cu` must accept a `cudaStream_t` argument and pass it to all kernel
launches and async memcpy calls:
```cpp
void PointGroupGPU::solve_closed(..., cudaStream_t stream = 0) {
  gpu_compute_functions<<<grid, block, 0, stream>>>(...);
  // ... all kernel launches take stream argument
}
```

### Critical: Accumulation Safety

The global accumulation of forces and RMM output currently happens via
`add_rmm_output` (CPU scatter) and `forces` accumulation. With streams, multiple
groups may finish concurrently and try to update shared global arrays.

**Solution**: Use `cudaStreamAddCallback` or per-group event + single-threaded
CPU reduction:
```cpp
// Option A: Single CPU thread collects results after all streams sync
cudaDeviceSynchronize();
for (auto& g : groups) g->reduce_output_to_global(...);

// Option B: GPU atomic scatter via optimize_rmm_gather_gpu scatter kernel
// (preferred: no CPU involvement in hot path)
```

`GlobalMemoryPool` is not thread-safe for concurrent streams — each group's memory
is pre-allocated before `solve()` is called (in `regenerate()`), so no runtime
allocation occurs during streaming. Safe if `resize()` is not called during `solve()`.

## Impact

| Scenario | NUM_STREAMS | Expected Speedup |
|---|---|---|
| Many small groups (M<20, P<128) | 4 | 1.5–2× for compute kernels |
| Mixed groups (some large, some small) | 2 | 1.2–1.5× |
| Few large groups (M>50, P>512) | 1 (no benefit) | <1.05× |
| Copy-compute overlap only | 2 | 1.1–1.3× |

Overall end-to-end estimate: **1.2–1.8×** for typical LIO workloads with mixed
group sizes. The original "1.5x–2x" estimate is achievable only for small-group-
dominated systems.

## Difficulty Assessment
**High** (accurate) — Reduced slightly from original since get_rmm_input GPU port
resolves the biggest serialization issue independently.

Correctness risks:
- Stream ordering: ensure H2D for group N+1 does not use buffers still being read
  by group N's kernels on a different stream. Use `cudaStreamWaitEvent` fences if
  groups share any buffers.
- `texture` binding: texture objects (`rmm_cuArray`) are group-private, so safe.
  After `optimize_density_texture.md`, textures are replaced with plain pointers —
  even safer.
- Fortran-side accumulation: `add_rmm_output` writes to `fortran_vars.rmm`. Multiple
  streams completing concurrently → CPU race. Fix: defer all reductions to after
  `cudaDeviceSynchronize()`.

## Files to Modify
- `g2g/partition.h`: Add `streams[]` member to `Partition`.
- `g2g/partition.cpp`: Stream creation, solve loop update, stream synchronization.
- `g2g/cuda/iteration.cu`: Add `cudaStream_t stream` parameter to `solve_closed` /
  `solve_opened`; propagate to all kernel launches and async copies.
- `g2g/common.h`: Define `NUM_STREAMS = 4` (or make runtime-configurable).

## Estimations
- Small-group-dominated systems: **1.5–2× speedup**.
- Typical mixed systems: **1.2–1.5× speedup**.
- Large-group-dominated: **<1.1× speedup** (not worth the complexity).
- Implementation: 3–5 days including stream synchronization safety validation.
