# Claude Notes — g2g

Subsystem-specific guidance for AI-assisted development of `g2g`.
See the **root `CLAUDE.md`** for project-wide build commands, environment setup,
running tests, and the CUDA build environment notes (nvcc path, GENCODE_FLAGS).
See `GEMINI.md` for a general architectural overview of this directory.

---

## Partition System — CPU/GPU Work Distribution

### Overview

The `Partition` object (global singleton `G2G::partition`) divides all DFT integration
points into `PointGroup` objects and distributes them across CPU and GPU workers.
The two key files are:

- `regenerate_partition.cpp` — builds the partition once at the start of each SCF cycle
- `partition.cpp` — drives the per-iteration `solve()` loop
- `partition.h` — `PointGroup`, `PointGroupCPU`, `PointGroupGPU`, `Partition` class definitions

### PointGroup types

| Class | `is_big_group()` | Worker | Condition |
|---|---|---|---|
| `PointGroupCPU<T>` | `false` | CPU thread | `points.size() <= SPLITPOINTS` (default 200) |
| `PointGroupGPU<T>` | `true` | GPU thread | `points.size() > SPLITPOINTS` |

`SPLITPOINTS` is read from env `LIO_SPLIT_POINTS` (default 200) in `regenerate_partition.cpp`.
`MINCOST` (env `LIO_MINCOST_OFFSET`, default 250000) is an additive offset in the cost function.

### `regenerate_partition.cpp` — partition build (called once per SCF start)

1. **Bounding prism** computed from atom positions + max basis function radii.
2. **Points generated** for each atom's radial shells × angular grid. Points inside the
   bounding prism are sorted into a 3-D array of cubes (cell size = `little_cube_size = 8 Å`)
   or into per-atom spheres (inner shells go to sphere if within `sphere_radius` fraction).
3. **PointGroup objects created**: each non-empty prism cell → one `PointGroupCPU` or
   `PointGroupGPU` depending on point count vs. `SPLITPOINTS`.
4. **Basis functions assigned**: `assign_functions_as_cube()` / `assign_functions_as_sphere()`
   determines which of the M global basis functions overlap with each group.
5. **Weights computed**: Becke partitioning weights per integration point.
6. **Work bins built** (`Partition::compute_work_partition()`):
   - **CPU groups**: bin-packed into `cpu_threads` bins using a binary-search on capacity to
     minimize the maximum bin cost. Cost = `10 * (npts * M_group * (1 + M_group)) / 2 + MINCOST`.
   - **GPU groups**: round-robin assigned to `gpu_threads` GPU bins (`work[cpu_threads..]`).
   - Result: `work[i]` = list of group indices for thread `i`.

### `Partition::solve()` — per-iteration execution (called every SCF iteration)

```
Thread layout: num_threads = cpu_threads + gpu_threads
               schedule(static) — each thread gets exactly 1 work bin
Thread 0 … cpu_threads-1 → CPU bins (PointGroupCPU groups, solved on CPU)
Thread cpu_threads       → GPU bin 0 (PointGroupGPU groups, solved on GPU)
```

**Per thread, inner loop** (serial within each thread's bin):
```
for j in work[i]:
    group[j]->solve_closed() / solve_opened()
    timeforgroup[j] = elapsed
next[i] = total thread time
```

**After the parallel region**, CPU merges per-thread outputs:
- `fort_forces_ms[i]` → accumulate into global forces array
- `rmm_outputs[i]` → accumulate into global RMM density matrix
- `rebalance()` — adjusts `work[]` bins for the next iteration based on actual `timeforgroup[]`

### GPU group execution sequence (one `PointGroupGPU::solve_closed()` call)

```
1. compute_functions()          → launch transpose kernels on persistent non-blocking
                                   streams (transpose_stream_1/2), return immediately
                                   (skipped when fgm=-1 caching is active after iter 1)
2. gpu_gather_rmm()             → GPU kernel: gather global packed RMM into local submatrix
                                   (shared global RMM buffer uploaded once per iteration, epoch-gated)
3. cudaMemcpy2DToArrayAsync()   → D2D copy from flat GPU buffer to CUDA array for texture reads
                                   implicitly waits for transpose streams (blocking stream rule)
4. Launch density/rmm kernels   → stream 0
5. gpu_scatter_rmm()            → GPU kernel: atomicAdd local Fock into global packed Fock buffer
                                   (no sync needed — next group's work queues behind on stream 0)
→ then step 1 for next group
6. (after ALL groups) sync + D2H download of global Fock buffer (once per iteration)
```

**Key design**: Steps 2 and 5 (gather/scatter) run entirely on GPU, eliminating the per-group
CPU serialization that previously required `cudaStreamSynchronize` + CPU scatter between every
group. The global Fock buffer (`s_global_fock_dev`) is zeroed once per SCF iteration
(epoch-gated) and accumulated via `atomicAdd(double)` (natively supported on SM 6.0+).

### Measured configuration (fosfatoQMMM, 34 QM atoms, 25 SCF iters)

*Last profiled: 2026-03-23, after all optimizations including fgm=-1 caching, AINT float, GPU scatter.*

| Parameter | Value |
|---|---|
| cpu_threads | 15 (= OMP_NUM_THREADS − 1 GPU) |
| gpu_threads | 1 |
| SPLITPOINTS | 200 (default) |
| GPU groups per SCF iter | ~45 (= 1134 gpu_compute_density calls / 25) |
| **Wall time (total)** | **3.22 s** |
| Post-SCF (AINT float, one-time) | ~418 ms |
| GPU kernel time (total) | 1.34 s (42% of wall) |
| GPU kernel time (SCF only) | 0.92 s (29% of wall) |
| Memcpy + memset | 40 + 5 = 45 ms (<2% of wall) |
| cudaMalloc + cudaFree | 92 ms (3% of wall) — mostly first-iter + AINT |
| cudaStreamSynchronize | 837 ms / 151 calls (forces sync only, no RMM sync) |

### GPU kernel time breakdown (fosfatoQMMM)

**SCF kernels (called per iteration × 25 iters):**

| Kernel | Total time | % GPU | Calls | Avg/call |
|---|---|---|---|---|
| gpu_compute_density (GGA) | 602 ms | 45.0% | 1134 | 531 µs |
| gpu_update_rmm | 193 ms | 14.4% | 1050 | 183 µs |
| gpu_compute_density_derivs | 75 ms | 5.6% | 42 | 1.79 ms |
| gpu_compute_forces | 12 ms | 0.9% | 42 | 276 µs |
| transpose\<vec4\> | 7 ms | 0.5% | 84 | 83 µs |
| gpu_accumulate_point (all) | 4 ms | 0.3% | 1134 | 3.5 µs |
| gpu_compute_functions | 4 ms | 0.3% | 42 | 87 µs |
| gpu_gather_rmm | 3 ms | 0.2% | 1134 | 2.9 µs |
| gpu_compute_weights | 3 ms | 0.2% | 42 | 69 µs |
| gpu_scatter_rmm | 3 ms | 0.2% | 1050 | 2.4 µs |
| transpose\<float\> | 1 ms | 0.1% | 42 | 20 µs |

Note: with `fgm=-1` caching, `compute_functions` and transpose run only on the first
iteration (42 calls = 42 groups × 1 iter). Subsequent iterations reuse cached values.

**Post-SCF (AINT float, called once after convergence):**

| Kernel | Total time |
|---|---|
| gpu_qmmm_forces (all angular momenta) | 268 ms |
| gpu_coulomb_forces (all angular momenta) | 104 ms |
| gpu_qmmm_fock (all angular momenta) | 46 ms |

### CUDA API overhead

| API call | Time | Calls | Note |
|---|---|---|---|
| cudaStreamSynchronize | 837 ms | 151 | Forces sync only (RMM scatter is async) |
| cudaDeviceSynchronize | 131 ms | 6 | Post-SCF barriers |
| cudaLaunchKernel | 69 ms | 5836 | — |
| cudaFree | 65 ms | 1240 | Residual (first-iter + AINT) |
| cudaMemcpy | 40 ms | 624 | Mostly AINT (pageable) |
| cudaMalloc | 27 ms | 1240 | Residual (first-iter + AINT) |

### Thread runtime: who is the bottleneck?

- **GPU thread** is the critical path. With `fgm=-1` caching + GPU scatter, most
  per-group CPU overhead is eliminated. The dominant cost is now the GPU kernels
  themselves (density + rmm_update) plus the 151 remaining `cudaStreamSynchronize`
  calls for forces readback.
- **CPU threads** (15 threads × many small groups) finish well within the GPU thread's time.

### Known optimizations applied

| Commit | Change | Effect |
|---|---|---|
| `ac87eef0` | Warp shuffle reductions in energy.h / energy_open.h | Smem 2560→256 B (LDA), 100% occupancy; preserves FP order |
| `21758bcb` | Persistent `transpose_stream_1/2` per PointGroupGPU | −345 ms streamSync, −22 ms create/destroy; −330 ms wall |
| (pinned) | Pinned host memory for all PointGroupGPU transfer buffers | cudaMemcpyAsync 1636→7.5 ms (218×); −2.2% wall |
| `e4a43707` | BLAS optimizations in converger_subs (DGEMM, DDOT) | Reduced Fortran CPU time |
| `5a7d1743` | BLAS in int3lu (DGEMV, DSPMV, DDOT) | Reduced per-iteration Fortran CPU |
| `104f8dc9` | DGELSS in DIIS solver (replaces DGELS) | Robust to float32 noise; bounded coefficients |
| (uncommitted) | Auto-detect GPU memory caching (`fgm=-1`) | −89% malloc calls, −93% malloc+free time; **−34% wall** (5.87→3.87s) |
| `627e32b5` | AINT float precision (`aint_mp=1` default) | AINT kernels 1.65× faster; −15% wall (3.87→3.29s) |
| (uncommitted) | GPU-side RMM scatter (`gpu_scatter_rmm` + `gpu_gather_rmm`) | Eliminates per-group cudaStreamSync + CPU scatter; −5% wall (3.38→3.22s) |

### GPU memory caching (`free_global_memory`)

The `free_global_memory` (fgm) parameter controls caching of basis function values across
SCF iterations. By default (`fgm=0.0`), no caching occurs — every iteration recomputes and
reallocates function/gradient/hessian buffers for all GPU groups (19K malloc/free calls,
2.07s overhead on fosfatoQMMM).

**Usage:**
- `free_global_memory = -1` — **auto-detect**: computes optimal cache budget from actual
  GPU memory availability and total cache needs. 20% headroom reserved for temporaries.
- `free_global_memory = 0.0` — no caching (default, backward-compatible, deterministic)
- `free_global_memory = 0.8` — use 80% of free GPU memory for caching (legacy manual mode)

**Measured (fosfatoQMMM, auto-detect):**

| Metric | fgm=0.0 | fgm=auto | Improvement |
|---|---|---|---|
| Wall time | 5.87 s | 3.87 s | **−34%** |
| cudaMalloc calls | 18,780 | 2,051 | −89% |
| cudaFree calls | 18,780 | 2,051 | −89% |
| malloc+free time | 2.07 s | 142 ms | −93% |
| gpu_compute_functions | 1,890 calls | 70 calls | −96% (first iter only) |
| transpose kernels | 3,920 calls | eliminated | −100% (after iter 1) |

**Nondeterminism warning**: Caching introduces run-to-run energy variation (~0.0002 Ha)
because the timing-dependent `rebalance()` function makes different group-to-thread
assignment decisions when the GPU finishes faster. This is NOT a caching correctness bug —
it's inherent to the timing-dependent rebalancer interacting with FP accumulation order.
See `todo/gpu/optimize_memory_pool.md` for details.

### Open optimization opportunities (ranked by expected impact)

1. ~~**GPU-side RMM gather/scatter**~~ — **DONE** (2026-03-23). `gpu_gather_rmm` +
   `gpu_scatter_rmm` replace CPU `get_rmm_input()` / `add_rmm_output()`. Eliminates
   per-group `cudaStreamSynchronize` for RMM (2030→151 calls). Actual speedup was ~5%
   (not the estimated 20-40%) because `fgm=-1` caching had already reduced the number
   of groups needing scatter, shrinking the overhead that scatter elimination targeted.
2. ~~**Reduce GlobalMemoryPool churn**~~ — **SOLVED** (2026-03-20). Auto-detect caching
   (`fgm=-1`) eliminates 89% of malloc/free calls. See "GPU memory caching" section above.
   Remaining 1,240 calls are from first-iter setup and AINT.
3. ~~**Replace tex2D with `__ldg`**~~ — **REJECTED** (2026-03-20). Causes 36% regression
   in `gpu_compute_density` on Pascal SM 6.1 due to loss of 2D spatial locality in texture
   cache (82.85% → 76.48% L1 hit rate). See `todo/gpu/optimize_density_texture.md` and
   `cuda/CLAUDE.md` for full analysis. Do NOT re-attempt on Pascal hardware.
4. **Eliminate forces cudaStreamSynchronize** — 837 ms in 151 calls. Currently each group
   syncs to read back forces. Could use GPU-side force accumulation (similar to Fock scatter)
   to eliminate per-group sync entirely. Expected ~10-20% wall time reduction.
5. **Multi-stream GPU pipeline** — launch group N+1 while N's kernels run.
   Now feasible since scatter is GPU-side, but diminishing returns with `fgm=-1` caching
   (fewer groups per iteration, kernels dominate).
6. **Open-shell GGA register reduction** — 93 regs → 56 regs (see TODO file).
7. **Dynamic OpenMP tasks for CPU** (`todo/cpu/optimize_cpu_threading.md`).

---

## SCF Convergence and Numerical Precision — Lessons Learned

**DO NOT add Kahan compensated summation (or any FP-order-changing optimization)
to GPU kernels whose output feeds into the DIIS convergence loop.** This was
extensively tested in March 2026 and consistently caused convergence regressions.

### Background

LIO uses hybrid float32/float64 precision: GPU kernels compute in float32,
results are cast to double on the CPU side. The SCF loop converges when
`rho_diff < 1e-6`. The DIIS accelerator (Pulay) extrapolates from stored Fock
matrices to predict the next iterate.

### What was tried and failed

| Change | Effect | Why it fails |
|---|---|---|
| Kahan summation in `gpu_compute_density` (energy.h) bj-loop | 25 → 31 SCF iterations (fosfatoQMMM) | Changes float32 values → different DIIS trajectory → oscillation near threshold |
| Kahan in energy.h + `__launch_bounds__(64, 16)` | 25 → 31 iters, amplified | `__launch_bounds__` forces different register allocation → different FP instruction scheduling → compounds the effect |
| Kahan in `gpu_update_rmm` (rmm.h) inner loop | Fe3H2O6 open-shell energy error 0.004 Ha (threshold 1.5e-4) | Same mechanism: changed float32 Fock matrix → different DIIS path |
| Reducing `ndiis` from 30 to 8 | Convergence stalls at ~2e-6, never reaches 1e-6 | With float32 noise floor ~2e-6, DIIS needs MORE history vectors to occasionally find an extrapolation that pushes below threshold |

### Root cause: DIIS sensitivity to float32 noise

The density kernel (`gpu_compute_density`) sums only ~43–86 terms per thread.
Kahan improves precision from ~5e-7 to ~6e-8 relative error — but the values
are **numerically different** from the non-Kahan baseline. These different XC
contributions feed into DIIS, which builds a least-squares extrapolation from
stored Fock matrices. Near convergence (rho_diff ~1e-6), the DIIS trajectory is
exquisitely sensitive to the exact float32 noise pattern. A "more precise" noise
pattern is NOT necessarily a better one for convergence — it's just different,
and the baseline's noise pattern happened to produce a favorable DIIS trajectory.

### Key findings

1. **Float32 noise floor is ~2e-6 in rho_diff.** This is inherent to the hybrid
   precision architecture. Any change that shifts float32 values (even toward
   more precise ones) can move the effective noise floor enough to disrupt DIIS.

2. **`__launch_bounds__` changes FP results.** By forcing different register
   allocation, nvcc may reorder FMA/multiply/add instructions, producing
   bit-different float32 results. This alone moved convergence from 31→28 iters.

3. **ndiis=30 is necessary** (unlike literature's typical 6–12). With float32
   noise near 1e-6, DIIS needs a large vector history to find linear
   combinations that push rho_diff below threshold. Reducing to 8 vectors was
   catastrophic.

4. **Warp shuffle reductions are safe** IF the FP summation order matches the
   original volatile shared-memory pattern (cross-warp pair first, then
   intra-warp tree). See commit `ac87eef0` for the correct implementation.

### Guidelines for future precision work

- **Safe optimizations**: Structural changes that preserve FP order (warp
  shuffles matching old volatile order, shared memory layout, loop unrolling
  without reordering). These don't change numerical results.

- **Unsafe optimizations**: Kahan summation, double-precision accumulators in
  GPU kernels, `__launch_bounds__`, any change to the order of FP operations
  in kernels feeding DIIS. These change float32 bit patterns and will likely
  shift SCF convergence.

- **The real fix for precision**: Move the full density/Fock pipeline to
  float64 on GPU (requires `precision=1` build flag, `FULL_DOUBLE` macro).
  This eliminates the float32 noise floor entirely. Half-measures (Kahan in
  one kernel but not others) create precision mismatches that are worse than
  consistent float32.

- **Always run the full E2E test suite** (`cd test && ./new_tests.py`) after
  any kernel change, even "precision-only" ones. The fosfatoQMMM test
  (closed-shell, 25 iters) and Fe3H2O6 test (open-shell, restart) are both
  sensitive to float32 changes.

---
