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
