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

## Current Priorities (as of 2026-04-17)

Ranked by expected wall-time impact on the fosfatoQMMM benchmark.

**Profiling breakdown** (fosfatoQMMM, RTX 3080 Ti, warm wall **~1.84s**, internal 1.72s, 25 SCF iters):

| Phase | Time | % Wall | Notes |
|-------|------|--------|-------|
| Iteration × 25 (Fock+Diag+DIIS) | 1.19s | 65% | 47.6ms/iter; int3lu (268ms) + g2g solve (498ms) sequential |
| Initialize SCF | 0.44s | 24% | int3mem 132ms, XC grid 147ms, guess 55ms, 1-e Fock 31ms, G matrix 9ms |
| Forces | 0.12s | 7% | |
| Other (I/O, finalize) | ~0.09s | 5% | |

**Top per-iter cost**: `int3lu` (10.7ms/iter CPU) → `g2g solve` (~20ms/iter GPU+CPU partition) is **strictly sequential**. Running int3lu concurrent with g2g's GPU portion is the largest remaining lever.

| # | Optimization | Area | Expected Impact | File |
|---|---|---|---|---|
| 0 | **Enable cuda=2 build** | Build | 50-90ms estimated (free; activates existing `cublasmath/` paths) | [fortran/full_scf_port_evaluation.md](fortran/full_scf_port_evaluation.md) |
| 1 | Overlap int3lu ↔ g2g solve | Fortran/GPU | 150-270ms (6-15% wall) | [fortran/overlap_int3lu_g2g.md](fortran/overlap_int3lu_g2g.md) |
| ~~2~~ | ~~GPU arena allocator~~ — **DONE 2026-04-24** (async pool, −85.8 ms API, −5% wall) | Infrastructure | — | [infrastructure/optimize_gpu_allocator_async_pool_2026_04_24.md](infrastructure/optimize_gpu_allocator_async_pool_2026_04_24.md) |
| 3 | Density as GEMM | GPU | 100-200ms (5-10% wall); density is ~46% of GPU time | [gpu/optimize_density_gemm.md](gpu/optimize_density_gemm.md) |
| 4 | Partition auto-tune caching | Infrastructure | High for MD runs (one-time for single-point) | _(no research doc yet)_ |
| 5 | Open-shell GGA register reduction | GPU | Open-shell systems only (zero effect on fosfato) | [gpu/optimize_open_shell_registers.md](gpu/optimize_open_shell_registers.md) |

**Iteration reduction is not a lever on this workload** — LIO's 25-iter
convergence is near the float32 DIIS floor. See
[convergence/initial_guess_evaluation.md](convergence/initial_guess_evaluation.md).

**int3lu GPU offload** and **full SCF port to g2g** both evaluated and
rejected/deferred — see rejected table below for the decision rationale.

## Key Completed Work

| Optimization | Measured Result | File |
|---|---|---|
| Warp shuffle reductions | 10x smem reduction, 100% LDA occupancy | [gpu/optimize_warp_shuffle.md](gpu/optimize_warp_shuffle.md) |
| GPU memory auto-caching (fgm=-1) | **34% wall time speedup** | [gpu/optimize_memory_pool.md](gpu/optimize_memory_pool.md) |
| GPU-side RMM gather/scatter | 5% wall time | [gpu/optimize_rmm_gather_gpu.md](gpu/optimize_rmm_gather_gpu.md) |
| AINT float precision | 15% wall time | _(commit 627e32b5, no research doc)_ |
| DGELSS solver (DIIS fix) | Robust to float32 noise | [convergence/converger_optimizations.md](convergence/converger_optimizations.md) |
| Pinned host memory | cudaMemcpyAsync 218x faster | [infrastructure/optimize_pinned_memory.md](infrastructure/optimize_pinned_memory.md) |
| Fortran allocation hoisting | Eliminated per-iter heap churn in Dens_build, DSYEVD, int3lu | [fortran/blas_optimization.md](fortran/blas_optimization.md) |
| Converger direct P'_ON from eigenvectors | Eliminated 2 DGEMMs/iter in converger (commit 7325eeb3) | _(commit only)_ |
| int3mem OpenMP parallelization | Coulomb precalc 418ms → 132ms (286ms saved, commit 7229cad3) | _(commit only)_ |
| Cholesky for G matrix | dgesdd+dsytrf/i → dpotrf/i; 125ms → 9ms (commit de2d2f21) | _(commit only)_ |
| GPU async memory pool | 1298 cudaMalloc+Free 100.7ms → 14.9ms; −5% wall | [infrastructure/optimize_gpu_allocator_async_pool_2026_04_24.md](infrastructure/optimize_gpu_allocator_async_pool_2026_04_24.md) |

## Rejected / Dead Ends / Diminished Returns

Don't re-investigate these without reading the analysis first.

| Investigation | Why Rejected | File |
|---|---|---|
| tex2D → `__ldg` | 36% on Pascal (cache miss), 2.5× on Ampere (address compute overhead) | [gpu/optimize_density_texture.md](gpu/optimize_density_texture.md) |
| FP16 / TF32 mixed precision | No Tensor Cores on Pascal | [gpu/optimize_mixed_precision.md](gpu/optimize_mixed_precision.md) |
| Kahan compensated summation | Disrupts DIIS convergence trajectory | [convergence/INDEX.md](convergence/INDEX.md) |
| Level shifting | Minimal benefit for LIO's systems | [convergence/level_shifting_evaluation.md](convergence/level_shifting_evaluation.md) |
| Dual-row → single-row restructuring | Float32 reduction non-associativity; proven impossible | [convergence/INDEX.md](convergence/INDEX.md) |
| Forces cudaStreamSync elimination | 837ms is real GPU work, only 5.5ms eliminable (0.2% wall) | [gpu/async_execution.md](gpu/async_execution.md) |
| Multi-stream GPU pipeline | CPU launch 60µs vs 700µs kernel; GPU never starved | [gpu/stream_sharding.md](gpu/stream_sharding.md) |
| Dynamic OpenMP tasks | CPU threads already balanced (13ms vs 18ms GPU); idle only 4ms/iter | [cpu/optimize_cpu_threading.md](cpu/optimize_cpu_threading.md) |
| Aufbau initial guess | 26-27 iters vs baseline 25; slower despite better starting energy. LIO 1-e guess + DIIS already tuned | [convergence/initial_guess_evaluation.md](convergence/initial_guess_evaluation.md) |
| int3lu GPU offload | Same ceiling as CPU/GPU overlap (267ms) at 3-5× the implementation cost. Dominated by overlap | [fortran/int3lu_gpu_offload_evaluation.md](fortran/int3lu_gpu_offload_evaluation.md) |
| Full SCF port to g2g | High absolute ceiling (440-565ms) but poor ROI (11-14 weeks); incremental stack delivers similar savings in 4-6 weeks. Revisit at M≥1500 or MD throughput | [fortran/full_scf_port_evaluation.md](fortran/full_scf_port_evaluation.md) |
