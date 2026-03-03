# Optimization: CUDA Graphs and Launch Optimization

## Summary
`solve_closed` in `g2g/cuda/iteration.cu` performs a fixed-topology sequence of
kernels and async memcpys per point group: compute_functions → transpose →
gpu_compute_density → gpu_accumulate_point → [density_derivs → forces] →
gpu_update_rmm. Each kernel launch incurs CPU-side driver overhead (~5–10 µs per
launch on CUDA 12). For small groups where kernel execution is <50 µs, launch
overhead can represent 10–30% of total wall time.

CUDA Graphs (available since CUDA 10, fully stable on CUDA 12) capture this sequence
once and replay it with a single API call, amortizing launch overhead.

## Practical Constraints and Limitations

**For Molecular Dynamics simulations (the primary use case of LIO), CUDA Graphs
have limited applicability:**

1. Every MD step, atomic positions change → `compute_functions` is re-run with
   new grid points → buffer contents change (but not sizes or pointers for a given group).
2. Kernel launch parameters (grid/block dimensions) depend on `number_of_points` and
   `group_m` which are FIXED per group after `regenerate()`. Pointers do not change
   within an MD run unless the group is re-partitioned.
3. `cudaMemcpy2DToArrayAsync` to texture arrays IS capturable since CUDA 11.1.
4. The `get_rmm_input()` CPU work cannot be captured — it runs on the host.
   If moved to GPU (see `optimize_rmm_gather_gpu.md`), the full sequence becomes
   capturable.

**Conclusion**: CUDA Graphs are viable AFTER implementing `optimize_rmm_gather_gpu.md`.
Until then, the CPU stall in `get_rmm_input` would break the graph capture anyway.

## Proposal

### Prerequisite: GPU-side RMM gather (see `optimize_rmm_gather_gpu.md`)

### Implementation

1. **Add graph state to `PointGroupGPU`** (`g2g/partition.h`):
   ```cpp
   cudaGraph_t     cuda_graph     = nullptr;
   cudaGraphExec_t cuda_graph_exec = nullptr;
   bool            graph_valid    = false;
   ```

2. **Capture on first invocation** (`g2g/cuda/iteration.cu`):
   ```cpp
   if (!graph_valid) {
     cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
     // ... all kernel launches and async memcpys ...
     cudaStreamEndCapture(stream, &cuda_graph);
     cudaGraphInstantiate(&cuda_graph_exec, cuda_graph, nullptr, nullptr, 0);
     graph_valid = true;
   }
   cudaGraphLaunch(cuda_graph_exec, stream);
   ```

3. **Invalidation trigger**: Set `graph_valid = false` when group is re-partitioned
   (i.e., when `regenerate()` is called). The next `solve()` re-captures.

4. **`cudaGraphExecUpdate` instead of re-instantiation**: If only kernel parameters
   (not topology) change, use `cudaGraphExecUpdate` which is cheaper than full
   re-instantiation (~100 µs vs ~1 ms).

## Impact
- **Small groups** (P < 256, M < 20): kernel launch overhead >20% of time.
  Graph replay gives **1.2–2× speedup on the RMM-heavy path**.
- **Large groups**: launch overhead is negligible. **<5% benefit**.
- **Net overall**: **5–15% improvement** in systems with many small groups (large
  molecules partitioned into many small cubes).
- Best for: geometry optimization (many iterations, fixed structure per step) >
  molecular dynamics (positions change, but group topology stays fixed).

## Difficulty Assessment
**Medium–High** (was listed as Medium; corrected)

- `cudaMemcpy2DToArrayAsync` to texture arrays requires CUDA 11.1+ for graph capture.
  Confirmed available with CUDA 12.0.
- The branching logic in `solve_closed` (LDA vs GGA, forces vs energy-only, open vs
  closed shell) means different graph topologies per configuration. Consider
  pre-capturing one graph per configuration or using conditional graph nodes (CUDA 12+).
- Timer events (`timers.xxx.start()`) insert `cudaEventRecord` into the stream; these
  ARE capturable. However, profiling tools may interact badly with graph capture.
  Add `#ifndef LIO_PROFILE` guards around timer calls during capture.

## Files to Modify
- `g2g/partition.h`: Add `cuda_graph*` members to `PointGroupGPU`.
- `g2g/cuda/iteration.cu`: Capture/launch logic around kernel sequence.
- `g2g/partition.cpp`: Invalidate graphs in `regenerate()`.

## Estimations
- Best case (many small groups, after GPU RMM gather): **10–20% overall speedup**.
- Typical case: **5–10%**.
- Worst case (few large groups): **<2%**.
- Implementation: 3–5 days including conditional graph topology handling.
