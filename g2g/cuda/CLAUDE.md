
## Profiling Procedures

### Hardware and tooling constraints

- **GPU**: GTX 1080 (SM 6.1, Pascal)
- **ncu (Nsight Compute) requires SM 7.0+** — cannot be used on this hardware
- **nsys (Nsight Systems)** works but `nvprof` gives more detailed kernel-level data on SM 6.x
- **nvprof** is the primary profiling tool for this project

### Reference test case

The standard profiling benchmark is **fosfatoQMMM** (34 QM atoms, closed-shell GGA, 25 SCF
iterations). Located at `test/LIO_test/03_fosfatoQMMM/`.

Input files: `fos.in`, `fos.xyz`, `basis`
Binary: `liosolo/liosolo`

### Step 1: Build

```bash
cd /media/manuel/storage/dev/lio
make clean && make cuda=1 cpu=0
```

For profiling, use the normal release build (not `dbg=1`). Debug builds with `-g -G` disable
optimizations and give misleading timings.

### Step 2: Measure wall time (without profiler overhead)

Always measure wall time **without** nvprof first, since nvprof adds ~10-20% overhead:

```bash
source liohome.sh
cd test/LIO_test/03_fosfatoQMMM
time ../../../liosolo/liosolo -i fos.in -c fos.xyz -b basis -v > /dev/null 2>&1
```

Expected output (as of 2026-03-19): `real ~5.8s`

### Step 3: Collect nvprof profile (save once, query many times)

**Critical**: save the profile to a `.nvvp` file and replay it. This avoids re-running the
full simulation for each different metric query.

```bash
source liohome.sh
cd test/LIO_test/03_fosfatoQMMM
nvprof -o /tmp/profile.nvvp ../../../liosolo/liosolo -i fos.in -c fos.xyz -b basis -v > /dev/null 2>&1
```

**Important**: `liosolo` must be called directly, NOT via `run.sh`. nvprof does not profile
child processes by default, so calling `./run.sh` (which spawns liosolo as a subprocess)
produces an empty profile. Use `--profile-child-processes` only if you must go through a
wrapper script.

### Step 4: Query the saved profile

All subsequent queries use `-i /tmp/profile.nvvp` (no re-execution):

#### GPU kernel summary (most useful — shows time per kernel)
```bash
nvprof -i /tmp/profile.nvvp --print-gpu-summary
```

Key columns: `Time(%)`, `Time`, `Calls`, `Avg`, `Name`

This tells you which kernels dominate GPU time. Look for:
- `gpu_compute_density` — should be #1 (density accumulation)
- `gpu_update_rmm` — Fock matrix update
- `transpose<vec4>` — data layout transformations
- `gpu_compute_functions` — basis function evaluation
- `gpu_qmmm_forces` / `gpu_coulomb_forces` / `gpu_qmmm_fock` — post-SCF (called once)

#### CUDA API summary (shows CPU-side overhead)
```bash
nvprof -i /tmp/profile.nvvp --print-api-summary
```

Key things to look for:
- `cudaMalloc` + `cudaFree` — GlobalMemoryPool churn (should be ~18K calls each)
- `cudaStreamSynchronize` — time CPU waits for GPU
- `cudaMemcpy` — host↔device transfers (high count = many small copies)
- `cudaMallocHost` — pinned memory allocations (should be ~350 if pinned optimization is active)
- `cudaMemcpyAsync` vs `cudaMemcpy` — async should dominate for SCF transfers

#### GPU trace (per-transfer details — pinned vs pageable)
```bash
nvprof -i /tmp/profile.nvvp --print-gpu-trace 2>&1 | head -20
```

The `SrcMemType` / `DstMemType` columns show `Pinned` or `Pageable` for each transfer.
To count:
```bash
nvprof -i /tmp/profile.nvvp --print-gpu-trace 2>&1 | grep -c "Pinned"
nvprof -i /tmp/profile.nvvp --print-gpu-trace 2>&1 | grep -c "Pageable"
```

SCF transfers (energy/forces/rmm) should show `Pinned`. AINT (post-SCF) transfers are
still `Pageable` and that's expected.

#### Hardware metrics (per-kernel deep-dive)
```bash
# Occupancy and stall reasons
nvprof -i /tmp/profile.nvvp --metrics achieved_occupancy,stall_exec_dependency,stall_sync \
    --kernels "gpu_compute_density"

# Memory metrics
nvprof -i /tmp/profile.nvvp --metrics tex_cache_hit_rate,l2_read_hit_rate \
    --kernels "gpu_compute_density"

# Warp efficiency
nvprof -i /tmp/profile.nvvp --metrics warp_execution_efficiency \
    --kernels "gpu_compute_density"
```

**Note**: `--metrics` queries may require re-running the program (nvprof replays kernels
with instrumentation). For metrics, you can also collect them during the initial run:
```bash
nvprof -o /tmp/profile_metrics.nvvp \
    --metrics achieved_occupancy,stall_exec_dependency,stall_sync,tex_cache_hit_rate \
    ../../../liosolo/liosolo -i fos.in -c fos.xyz -b basis -v > /dev/null 2>&1
```

But this makes the run much slower (~5-10×). Only do this for specific kernels you're
investigating, not as a routine step.

#### Register and shared memory usage (compile-time, no profiling needed)
```bash
cd /media/manuel/storage/dev/lio
make cuda=1 cpu=0 2>&1 | grep -E "registers|smem|shared"
```

Or add `--ptxas-options=-v` to NVCCFLAGS in `g2g/Makefile.cuda` to see per-kernel resource
usage at compile time.

### Step 5: Analyze results

#### Time budget framework

Parse the profile into these categories:

| Category | How to measure | What it means |
|---|---|---|
| GPU kernel time | `--print-gpu-summary`, sum all kernel times | Actual computation on GPU |
| GPU memcpy time | `--print-gpu-summary`, sum `[CUDA memcpy *]` rows | Data transfer time |
| cudaMalloc+cudaFree | `--print-api-summary`, sum those two rows | Memory management overhead |
| cudaStreamSync | `--print-api-summary` | CPU waiting for GPU |
| CPU overhead | wall_time − GPU_kernel − cudaMalloc/Free − sync | get_rmm_input + add_rmm_output + Fortran |

**Total GPU time** = first kernel's time / (first kernel's percentage / 100).
For example, if `gpu_compute_density` shows `37.1%` and `1.047s`, total = 1.047/0.371 = 2.82s.

#### SCF vs post-SCF split

SCF kernels are called many times (1960 density calls = 78 groups × 25 iters).
Post-SCF kernels (`gpu_qmmm_forces`, `gpu_coulomb_forces`, `gpu_qmmm_fock`) are called
once total (after convergence). Separate them when analyzing optimization targets:
- SCF kernel time / 25 = per-iteration GPU cost
- Post-SCF is fixed overhead regardless of convergence speed


