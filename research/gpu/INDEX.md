# GPU Kernel Optimizations

Research and implementation notes for CUDA kernel performance.
Current hardware: RTX 3080 Ti (SM 8.6 Ampere). Previous: GTX 1080 (SM 6.1 Pascal).
The density kernel (`gpu_compute_density`) dominates at 45% of GPU time — start there.

## Status Legend

- **DONE** — implemented and measured
- **REJECTED** — investigated and ruled out (read before re-attempting)
- **OPEN** — viable but not yet implemented

## Files

### Completed / Closed

| File | Status | Summary |
|------|--------|---------|
| [optimize_warp_shuffle.md](optimize_warp_shuffle.md) | DONE | `__shfl_down_sync` reductions: 10x smem reduction, 100% LDA occupancy |
| [optimize_memory_pool.md](optimize_memory_pool.md) | DONE | `fgm=-1` auto-caching: 34% wall time speedup, 89% fewer malloc/free |
| [optimize_rmm_gather_gpu.md](optimize_rmm_gather_gpu.md) | DONE | GPU-side RMM gather/scatter: eliminated per-group CPU sync, 5% speedup |
| [fgm_correctness_bug.md](fgm_correctness_bug.md) | FIXED | Weights buffer reuse bug in `compute_weights` (1-line fix) |
| [optimize_density_texture.md](optimize_density_texture.md) | REJECTED | tex2D vs `__ldg`: 36% regression on Pascal, 2.5× on Ampere (different root causes). Dead end. |
| [optimize_mixed_precision.md](optimize_mixed_precision.md) | REJECTED | FP16/TF32 not viable on Pascal (no Tensor Cores) |

### Open — High Impact

| File | Impact | Summary |
|------|--------|---------|
| [async_execution.md](async_execution.md) | DIMINISHED | Phases 1-2 DONE; remaining syncs are post-SCF only (~5.5ms/3.22s = 0.2%), not worth pursuing |
| [optimize_open_shell_registers.md](optimize_open_shell_registers.md) | HIGH | Open-shell GGA: 93 regs → 56 regs (34% → 56% occupancy) by splitting into 2 closed-shell calls |
| [stream_sharding.md](stream_sharding.md) | NOT WORTH IT | CPU launch overhead 60µs/group vs 700µs kernel; GPU never starved with fgm=-1 |
| [optimize_density_gemm.md](optimize_density_gemm.md) | HIGH | Only remaining >10% opportunity for density kernel; eliminates texture+divergence; FP-order risk |
| [optimize_rmm.md](optimize_rmm.md) | MEDIUM | Replace custom RMM SYRK with cuBLAS `cublasSsyrk` |

### Open — Lower Impact / Speculative

| File | Impact | Summary |
|------|--------|---------|
| [optimize_energy_derivs.md](optimize_energy_derivs.md) | MEDIUM | Force kernel O(M²·P) bottleneck; GEMM reformulation |
| [optimize_kernel_fusion.md](optimize_kernel_fusion.md) | NOT WORTH IT | accumulate_point is 3.5µs/call (0.3% GPU); fusion would push regs to ~80 → 25% occupancy |
| [optimize_screening.md](optimize_screening.md) | MEDIUM | Spatial block culling for basis functions (high difficulty) |
| [optimize_transpose.md](optimize_transpose.md) | LOW | Eliminate transpose by writing compute_functions in transposed layout |
| [optimize_weight_cache.md](optimize_weight_cache.md) | LOW | Cache Becke weights across SCF (depends only on atom positions) |
| [optimize_cuda_graphs.md](optimize_cuda_graphs.md) | LOW | CUDA Graphs: limited applicability (topology changes per MD step) |
| [optimize_kernel_batching.md](optimize_kernel_batching.md) | LOW | Batch small groups in single launch; limited benefit on 20-SM GPU |
| [optimize_occupancy_tuning.md](optimize_occupancy_tuning.md) | REF | Block size / register analysis for SM 6.1 |

### Reference / Analysis

| File | Summary |
|------|---------|
| [roofline_gpu_compute_density.md](roofline_gpu_compute_density.md) | Full roofline + opportunity analysis: kernel is at 52-69% of realistic ceiling; GEMM reformulation is only remaining >10% path |
