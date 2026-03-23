# Infrastructure and Cross-Cutting Optimizations

General optimizations that span the CPU/GPU boundary: memory management,
data layout, threading, and the partition rebalancer.

## Files

| File | Status | Summary |
|------|--------|---------|
| [optimize_pinned_memory.md](optimize_pinned_memory.md) | DONE | Pinned host memory for all iteration buffers (cudaMemcpyAsync 218x faster) |
| [rebalance_cpu_gpu.md](rebalance_cpu_gpu.md) | REF | Rebalancer analysis: throughput impact, timing-dependent nondeterminism |
| [optimize_gpu_allocator.md](optimize_gpu_allocator.md) | OPEN | True GPU memory pool/arena allocator to reduce cudaMalloc latency |
| [optimize_matrix_move.md](optimize_matrix_move.md) | OPEN | Move semantics for HostMatrix/CudaMatrix to eliminate deep copies |
| [optimize_matrix_resize.md](optimize_matrix_resize.md) | OPEN | Capacity tracking in HostMatrix::resize to avoid reallocation |
| [optimize_soa_layout.md](optimize_soa_layout.md) | OPEN | Convert PointGroup from AoS to SoA for SIMD efficiency |
| [optimize_parallel_regenerate.md](optimize_parallel_regenerate.md) | OPEN | Parallelize grid generation with OpenMP (benefits MD steps) |
