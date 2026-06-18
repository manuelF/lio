# Claude Notes — g2g

Subsystem-specific guidance for AI-assisted development of `g2g`.
See the **root `CLAUDE.md`** for project-wide build commands, environment setup,
running tests, and the CUDA build environment notes (nvcc path, GENCODE_FLAGS).

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

### Thread runtime: who is the bottleneck?

- **GPU thread** is the critical path. With `fgm=-1` caching + GPU scatter, most
  per-group CPU overhead is eliminated. The dominant cost is now the GPU kernels
  themselves (density + rmm_update) plus the 151 remaining `cudaStreamSynchronize`
  calls for forces readback.
- **CPU threads** (15 threads × many small groups) finish well within the GPU thread's time.

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

**Nondeterminism warning**: Caching introduces run-to-run energy variation (~0.0002 Ha)
because the timing-dependent `rebalance()` function makes different group-to-thread
assignment decisions when the GPU finishes faster. This is NOT a caching correctness bug —
it's inherent to the timing-dependent rebalancer interacting with FP accumulation order.
See `../research/gpu/INDEX.md` for the partition/caching optimization history.

### Background

LIO uses hybrid float32/float64 precision: GPU kernels compute in float32,
results are cast to double on the CPU side. The SCF loop converges when
`rho_diff < 1e-6`. The DIIS accelerator (Pulay) extrapolates from stored Fock
matrices to predict the next iterate.

### Guidelines for future precision work

- **Safe optimizations**: Structural changes that preserve FP order (warp
  shuffles matching old volatile order, shared memory layout, loop unrolling
  without reordering). These don't change numerical results.

- **Always run the full E2E test suite** (`make check`, or `cd test && ./new_tests.py`)
  after any kernel change, even "precision-only" ones. The fosfatoQMMM test
  (closed-shell, 25 iters) and Fe3H2O6 test (open-shell, restart) are both
  sensitive to float32 changes.

- **Run the kernel unit tests under `compute-sanitizer --tool racecheck`** after any
  change to `g2g/cuda/kernels/` (the per-kernel test binaries live in
  `test/unit_tests/kernels/`). Float32 shared-mem races don't produce visibly wrong
  output (32-bit stores are atomic), but the same race explodes in
  FULL_DOUBLE (64-bit stores tear). Racecheck is the only reliable detector.
