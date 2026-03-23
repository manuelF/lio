# LIO Research & Optimization Knowledge Base

Central index for all research, optimization notes, and technical investigations.
Each area has its own `INDEX.md` with per-file status and summaries — read the
relevant sub-index before diving into individual files.

## How to Use This

1. **Starting a new optimization?** Read [guides/](guides/INDEX.md) first — especially
   `cuda_optimization_guide.md` for the tier framework and profiling methodology.
2. **Modifying GPU kernels?** Read [convergence/](convergence/INDEX.md) key constraints
   before changing any kernel that feeds the SCF loop.
3. **Looking for what to work on next?** See the "Current Priorities" section below.

---

## Areas

| Area | Path | What's Inside |
|------|------|---------------|
| **GPU kernels** | [gpu/INDEX.md](gpu/INDEX.md) | 20 files — density, RMM, forces, transpose, memory caching. 6 completed, 14 open. |
| **CPU kernels** | [cpu/INDEX.md](cpu/INDEX.md) | 5 files — BLAS, threading, vectorization, screening. All open, lower priority. |
| **SCF convergence** | [convergence/INDEX.md](convergence/INDEX.md) | 4 files — DIIS solver, float32 noise, data flow analysis. Critical reading for kernel work. |
| **Infrastructure** | [infrastructure/INDEX.md](infrastructure/INDEX.md) | 7 files — memory management, data layout, rebalancer, pinned memory. |
| **Fortran (lioamber)** | [fortran/INDEX.md](fortran/INDEX.md) | 4 files — BLAS, modernization, testing, ECP. |
| **TD-DFT** | [tddft/INDEX.md](tddft/INDEX.md) | 1 file — Ehrenfest propagation optimization. |
| **Guides** | [guides/INDEX.md](guides/INDEX.md) | 2 files — master CUDA optimization guide, priority list. |

---

## Current Priorities (as of 2026-03-23)

Ranked by expected wall-time impact on the fosfatoQMMM benchmark (3.22s baseline).

| # | Optimization | Area | Expected Impact | File |
|---|---|---|---|---|
| 1 | Eliminate forces cudaStreamSynchronize | GPU | ~10-20% wall time | [gpu/async_execution.md](gpu/async_execution.md) |
| 2 | Open-shell GGA register reduction | GPU | 93→56 regs, 34%→56% occ | [gpu/optimize_open_shell_registers.md](gpu/optimize_open_shell_registers.md) |
| 3 | Multi-stream pipeline | GPU | Diminishing with fgm=-1 | [gpu/stream_sharding.md](gpu/stream_sharding.md) |
| 4 | Dynamic OpenMP tasks | CPU | Better load balancing | [cpu/optimize_cpu_threading.md](cpu/optimize_cpu_threading.md) |
| 5 | Density as GEMM | GPU | Reduce O(M²) to O(M) reads | [gpu/optimize_density_gemm.md](gpu/optimize_density_gemm.md) |

## Key Completed Work

| Optimization | Measured Result | File |
|---|---|---|
| Warp shuffle reductions | 10x smem reduction, 100% LDA occupancy | [gpu/optimize_warp_shuffle.md](gpu/optimize_warp_shuffle.md) |
| GPU memory auto-caching (fgm=-1) | **34% wall time speedup** | [gpu/optimize_memory_pool.md](gpu/optimize_memory_pool.md) |
| GPU-side RMM gather/scatter | 5% wall time | [gpu/optimize_rmm_gather_gpu.md](gpu/optimize_rmm_gather_gpu.md) |
| AINT float precision | 15% wall time | _(commit 627e32b5, no research doc)_ |
| DGELSS solver (DIIS fix) | Robust to float32 noise | [convergence/converger_optimizations.md](convergence/converger_optimizations.md) |
| Pinned host memory | cudaMemcpyAsync 218x faster | [infrastructure/optimize_pinned_memory.md](infrastructure/optimize_pinned_memory.md) |

## Rejected / Dead Ends

Don't re-investigate these without reading the analysis first.

| Investigation | Why Rejected | File |
|---|---|---|
| tex2D → `__ldg` | 36% regression on Pascal SM 6.1 | [gpu/optimize_density_texture.md](gpu/optimize_density_texture.md) |
| FP16 / TF32 mixed precision | No Tensor Cores on Pascal | [gpu/optimize_mixed_precision.md](gpu/optimize_mixed_precision.md) |
| Kahan compensated summation | Disrupts DIIS convergence trajectory | [convergence/INDEX.md](convergence/INDEX.md) |
| Level shifting | Minimal benefit for LIO's systems | [convergence/level_shifting_evaluation.md](convergence/level_shifting_evaluation.md) |
| Dual-row → single-row restructuring | Float32 reduction non-associativity; proven impossible | [convergence/INDEX.md](convergence/INDEX.md) |
