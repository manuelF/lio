# Full SCF Loop Port to g2g — Evaluation

**Status:** DEFERRED — high ceiling but poor ROI at M=364; revisit at M≥1500 or for MD throughput
**Date:** 2026-04-17

## The idea

Move the entire per-iteration SCF loop (currently Fortran: int3lu, converger
+ DIIS, base changes, Fock diagonalization, density build, convergence check)
into `g2g` C++/CUDA. Keep all matrices in device memory across iterations.
Fortran retains only I/O, configuration, and post-SCF analysis.

## Current partition

| Component | Time/iter | Size | Currently on | Cost if ported |
|-----------|-----------|------|--------------|----------------|
| `int3lu` | 10.7ms | cool ~tens MB, Pmat/Fmat M(M+1)/2 | CPU OpenBLAS | see [int3lu_gpu_offload_evaluation.md](int3lu_gpu_offload_evaluation.md) |
| `g2g_solve_groups(0, Ex, 0)` | 20ms | grid points | **Already GPU** | — |
| Base change AO→ON | ~1.5ms | 364² | CPU DGEMM (CUBLAS if cuda=2) | ~0.5ms |
| Converger (damping + DIIS) | 4.2ms | ndiis=30 × 364² | CPU | ~1-2ms |
| DSYEVD (diagonalize F') | 8.1ms | 364×364 eigenproblem | CPU LAPACK | cuSOLVER ~3ms |
| Base change ON→AO | ~1ms | 364² | CPU (or cumxp_r if cuda=2) | ~0.5ms |
| Density build (C·Cᵀ) | ~1ms | 364² | CPU DGEMM | ~0.5ms |
| Convergence check + misc | ~2ms | — | CPU | ~1ms |

**Total per-iter: 47.6ms** (measured), **theoretical GPU floor: ~20ms** (= g2g critical path).

## Existing GPU plumbing

LIO already has partial CUBLAS infrastructure, gated by `cuda=2`:
- `mask_cublas.f90` — `cublas_setmat`, persistent `dev_Xmat`/`dev_Ymat`
- `cublasmath/` — 12+ files wrapping cuBLAS: `cumxp_r`, `cumpx_r`, `cumxtf`,
  `basechange_cublas`, `commutator_cublas`, `cu_fock_commuts`
- `#ifdef CUBLAS` paths in `SCF.f90`, `converger_subs.f90`,
  `typedef_operator/BChange_data.f90`

**Today's build is `cuda=1`** (per `make cuda=1 cpu=1`), which disables all
of this. Rebuilding with `cuda=2` is a zero-code first step — measure that
before committing to a larger port.

## Ceiling analysis

Theoretical floor per iter ≈ max(g2g_solve, int3lu_gpu, diag_gpu, …) ≈ 20ms
(g2g remains critical path).

| Scenario | Per-iter | 25-iter total | Save vs 1190ms |
|----------|----------|---------------|----------------|
| Current (cuda=1) | 47.6ms | 1190ms | — |
| cuda=2 enabled, no new code | ~43-45ms estimated | ~1100ms | ~90ms (5%) |
| + overlap int3lu ↔ g2g | ~32-35ms | ~800ms | ~390ms (21%) |
| + density-as-GEMM + allocator | ~25-28ms | ~650ms | ~540ms (30%) |
| Full GPU SCF (theoretical) | ~20ms | 500ms | 690ms (38%) |
| Full GPU SCF (realistic) | ~25-30ms | 625-750ms | 440-565ms (24-31%) |

The "realistic" number reflects three penalties that hit LIO specifically at M=364:

1. **GPU launch overhead on small matrices.** cuSOLVER DSYEVD on 364² runs
   5-10 sub-kernels with sync points (~20-50µs each). Per-iter diag overhead
   alone: 100-500µs. DIIS inner products (30×30) are even worse. At M=364
   cuBLAS DGEMM hits ~50-70% of peak, not the 90%+ seen on M≥1024.

2. **int3lu scatter kernel.** Fmat accumulation via `kkind/kkinds` packed
   indexing cannot use cuBLAS — needs a custom kernel. See
   `int3lu_gpu_offload_evaluation.md` for the 3-4 week estimate.

3. **CUDA Graphs can't rescue this.** Graphs batch launches, but LIO's SCF
   topology varies (DIIS subspace grows, convergence branch). Graph captures
   would invalidate frequently. The NVIDIA-documented breakeven for Graphs
   is typically >10 kernels per graph AND stable topology — LIO has the
   former but not the latter.

## Effort estimate

| Phase | Work | Weeks |
|-------|------|-------|
| Enable cuda=2, measure CUBLAS gains | Rebuild, run e2e, compare timers | 0.2 |
| Port int3lu to GPU | Custom scatter + cuBLAS glue + validation | 3-4 |
| Migrate DIIS to device | Circular buffer on device, B-matrix solve via cuSOLVER | 2 |
| DSYEVD → cuSOLVER | Drop-in replacement, validate FP ordering | 1 |
| Persistent device state refactor | Extend `typedef_operator`, eliminate per-iter H2D/D2H | 2-3 |
| Numerical debugging (DIIS + float32) | The long tail | 3-4 |
| **Total** | | **11-14 weeks** focused |

## ROI comparison

| Approach | Realistic save | Effort | Wall-ms per week |
|----------|----------------|--------|------------------|
| Enable cuda=2 | 50-90ms | 1 day | 250-450 |
| Overlap int3lu ↔ g2g | 150-250ms | 1-2 weeks | 100-150 |
| GPU arena allocator | 60-90ms | 3-5 days | 60-90 |
| Density-as-GEMM | 100-150ms | 1-2 weeks | 70-100 |
| **Full SCF port** | **440-565ms** | **11-14 weeks** | **35-50** |

The full port has the highest **absolute** ceiling but the **worst** ROI per
engineer-week. The incremental stack (cuda=2 + overlap + density GEMM +
allocator) can deliver **360-580ms** in 4-6 weeks — comparable savings, 2-3×
the velocity, and each step is independently shippable/reversible.

## When the calculus flips

The full port's ceiling scales with:
- **System size.** At M ≥ 1500, launch overhead becomes a rounding error and
  cuBLAS/cuSOLVER approach peak. Per-iter floor drops from "realistic 25ms"
  toward the theoretical 20ms.
- **MD throughput.** Each MD step re-initializes SCF; with hundreds of steps,
  per-iter wall savings multiply linearly. A 30% per-SCF speedup on 10⁴ MD
  steps is multi-day throughput for production runs.
- **Open-shell / larger basis sets.** Doubles the per-iter BLAS volume;
  overlap gains vs. port gains diverge (overlap ceiling is bounded by
  min(int3lu, g2g); port ceiling keeps scaling).

**Revisit trigger:** if the target workload moves to M≥1500, or to MD runs
with >1000 steps, re-run this analysis. On fosfato at M=364 with single-point
SCF, the incremental stack is strictly better.

## First concrete step (if this direction is pursued)

Rebuild with `cuda=2` and measure. This activates the existing `cublasmath/`
infrastructure with **zero new code** and tells us how much of the
ceiling is already achievable:

```bash
make clean
make cuda=2 cpu=1
cd test/LIO_test/03_fosfatoQMMM && bash run.sh output
grep "Total time\|Convergence" output
```

If cuda=2 produces a measurable speedup, the next step is extending CUBLAS
paths into converger + diag. If it doesn't, the existing CUBLAS paths are
already bypassed and porting deeper components has an uncertain payoff.
