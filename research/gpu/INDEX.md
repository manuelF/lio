# GPU Kernel Optimizations

Research and implementation notes for CUDA kernel performance.
Current hardware: RTX 3080 Ti (SM 8.6 Ampere). Previous: GTX 1080 (SM 6.1 Pascal).

**2026-04-08 status:** On RTX 3080 Ti, g2g solve (all GPU+CPU kernels) is only
**11% of wall time** (0.39s / 3.63s on fosfatoQMMM). The bottleneck has shifted
to Fortran-side CPU work (converger, int3lu, DIIS). GPU kernel optimizations now
have diminishing returns for this system — a 2× density speedup saves ~3% wall.
Larger molecular systems will still benefit from GPU kernel work.

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
| [async_execution.md](async_execution.md) | CLOSED | Phases 1-2 DONE; remaining syncs are post-SCF only (~5.5ms = 0.2%), not worth pursuing |
| [optimize_open_shell_registers.md](optimize_open_shell_registers.md) | HIGH (open-shell only) | 93 regs → 56 regs (34% → 56% occ) by splitting into 2 closed-shell calls; no effect on closed-shell |
| [stream_sharding.md](stream_sharding.md) | CLOSED | CPU launch overhead 60µs/group vs 700µs kernel; GPU never starved with fgm=-1 |
| [optimize_density_gemm.md](optimize_density_gemm.md) | MEDIUM (was HIGH) | ~3% wall on fosfatoQMMM/3080Ti (g2g is only 11% of wall); larger impact on bigger systems |
| [optimize_rmm.md](optimize_rmm.md) | LOW (was MEDIUM) | cuBLAS SYRK; diminished by g2g being 11% of wall |

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
