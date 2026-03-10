# Claude Notes — g2g

Subsystem-specific guidance for AI-assisted development of `g2g`.
See the **root `CLAUDE.md`** for project-wide build commands, environment setup,
running tests, and the CUDA build environment notes (nvcc path, GENCODE_FLAGS).
See `GEMINI.md` for a general architectural overview of this directory.

---

## CUDA Kernel Unit Tests

Low-level tests for individual CUDA kernels live in `test/unit_tests/kernels/`.
They link against nothing in `g2g/` — each test includes the kernel header
directly inside `namespace G2G {}`, matching production usage in `iteration.cu`.

### Adding a new kernel test

1. Drop `<kernel_name>_test.cu` into `test/unit_tests/kernels/`.
2. Write the test:
   - Include kernel headers inside `namespace G2G { #include "../../../g2g/cuda/kernels/<kernel>.h" }`
   - Use `CUDA_CHECK(...)` and `test_utils::TestRunner` from `common/test_utils.h`
3. The shared `kernels/Makefile` picks it up automatically via `$(wildcard *_test.cu)`.

### Build overrides
```bash
# Force a specific GPU architecture
make GENCODE_FLAGS="-gencode arch=compute_80,code=compute_80 \
                    -gencode arch=compute_80,code=sm_80"
# Use a specific nvcc binary
make NVCC=/usr/local/cuda-12.0/bin/nvcc
```

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
2. get_rmm_input()              → CPU: gather RMM submatrix (~O(M_group²) work)
                                   OVERLAPS with transposes from step 1
3. cudaMemcpy2DToArrayAsync()   → upload RMM to CUDA texture array (stream 0)
                                   implicitly waits for transpose streams (blocking stream rule)
4. Launch density/rmm kernels   → stream 0
5. cudaStreamSynchronize(0)     → wait for all GPU work
6. add_rmm_output()             → CPU: scatter results back (~O(M_group²) work)
                                   GPU IS IDLE during this step
→ then step 1 for next group
```

**Critical idle gap**: Between groups, the GPU is idle during step 6 (`add_rmm_output`) and
any remaining setup before step 1 of the next group. With 76 groups × ~5.4 ms idle = ~410 ms
of GPU idle per SCF iteration (measured on fosfatoQMMM, 25 iters, wall time 12 s).

### Measured configuration (fosfatoQMMM, 34 QM atoms, 25 SCF iters)

| Parameter | Value |
|---|---|
| cpu_threads | 15 (= OMP_NUM_THREADS − 1 GPU) |
| gpu_threads | 1 |
| SPLITPOINTS | 200 (default) |
| GPU groups per SCF iter | ~76 (= 1890 gpu_compute_density calls / 25) |
| GPU kernel active time | ~100 ms / iter (~2500 ms total / 25 iters) |
| Wall time per iter | ~480 ms |
| GPU utilization | ~21% (kernel time / wall time) |
| GPU idle between groups | ~5.4 ms per group (= 410 ms / 76 groups) |

### Thread runtime: who is the bottleneck?

- **GPU thread** takes ~480 ms/iter (processes all 76 big groups sequentially).
- **CPU threads** (15 threads × many small groups) finish within ~480 ms (not the bottleneck).
- The GPU thread IS the critical path. GPU kernels only run 21% of GPU thread time; the
  remaining 79% is CPU overhead between launches (get_rmm_input + add_rmm_output).

### Known optimizations applied

| Commit | Change | Effect |
|---|---|---|
| `21758bcb` | Persistent `transpose_stream_1/2` per PointGroupGPU | `get_rmm_input` overlaps with transpose kernels; −345 ms streamSynchronize, −22 ms create/destroy overhead; −330 ms wall |

### Open optimization opportunities

1. **Eliminate add_rmm_output GPU idle gap** — biggest remaining opportunity. Options:
   - Async scatter: do add_rmm_output in a CPU worker thread while GPU starts next group
   - Fuse scatter into a CPU-side kernel (SIMD-vectorized gather/scatter)
2. **Multi-stream GPU pipeline** — launch group N+1 compute_functions *before* group N's
   add_rmm_output finishes (requires per-group output buffers, not shared rmm_outputs[i])
3. **Lower SPLITPOINTS** — move borderline groups to GPU (currently default 200 points).
   More GPU groups → higher GPU utilization, but more CPU overhead per group.
4. **Dynamic OpenMP tasks for CPU** (see `todo/cpu/optimize_cpu_threading.md`) —
   replace static bin-packing with `#pragma omp task` for automatic load balancing.
5. **Open-shell GGA register reduction** — 93 regs → 56 regs target (see TODO file).

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

## Kernel Notes

### transpose.h (`cuda/kernels/transpose.h`)

**Signature:**
```cpp
template <class T>
__global__ void transpose(T* odata, const T* idata, int width, int height);
```

**Inclusion:** must be inside `namespace G2G {}` (as in `cuda/iteration.cu`).

**Launch config:**
```cpp
dim3 block(TILE_DIM, BLOCK_ROWS);  // (32, 8)
dim3 grid((width + TILE_DIM-1) / TILE_DIM, (height + TILE_DIM-1) / TILE_DIM);
G2G::transpose<T><<<grid, block>>>(d_out, d_in, width, height);
```

**Memory layout:** input `height × width` row-major → output `width × height`
row-major with stride `height`:
```
output[j * height + i] == input[i * width + j]   // for all valid i, j
```

**Resource usage (SM 6.1, CUDA 12.0, `-O0 -G`):**
| Type | Registers | Shared mem |
|---|---|---|
| `float` | 19 | 4224 B (33 × 32 × 4 — the +1 bank-conflict padding) |
| `double` | 20 | 8448 B (33 × 32 × 8) |
