# Optimization: Occupancy Tuning via Block Size and Register Pressure Analysis

## Summary
The current kernel block sizes in `g2g/common.h` are hardcoded constants:
- `FUNCTIONS_BLOCK_SIZE = 128`
- `WEIGHT_BLOCK_SIZE = 128`
- `DENSITY_BLOCK_SIZE = 64`
- `DENSITY_DERIV_BLOCK_SIZE = 128`
- `RMM_BLOCK_SIZE_XY = 16` (16×16 = 256 threads)

These were likely set by intuition or early benchmarking. On SM 6.1 (Pascal), the
optimal block size depends on register usage per thread (which determines the maximum
concurrent warps per SM). None of the kernels use `__launch_bounds__` to communicate
register budget to the compiler, nor is `cudaOccupancyMaxPotentialBlockSize` used at
runtime to select optimal configurations.

GTX 1080 SM 6.1 limits per SM:
- Warps: 64 max (2048 threads)
- Blocks: 32 max
- Registers: 65536 per SM (max 255 per thread)
- Shared memory: 96 KB per SM (configurable: 48/64/96 KB L1 vs shared)

## Audit: Estimated Register Usage Per Kernel

| Kernel | Block size | Est. regs/thread | Warps/SM | Occupancy |
|---|---|---|---|---|
| `gpu_compute_functions` | 128 | ~40 | 32 | 50% |
| `gpu_compute_density` (LDA) | 64 | ~30 | 43 | 67% |
| `gpu_compute_density` (GGA) | 64 | ~55 | 23 | 36% |
| `gpu_compute_density_derivs` | 128 | ~35 | 37 | 58% |
| `gpu_update_rmm` | 256 (16×16) | ~25 | 32 | 50% |
| `gpu_compute_weights` | 128 | ~45 | 28 | 44% |

Note: Register estimates derived from CUDA resource analysis. Actual values require
`nvcc --ptxas-info` (add `--ptxas-options=-v` to NVCC flags to see at build time).

**The GGA density kernel at 36% occupancy is a significant concern** — it likely
limits throughput for GGA functionals (most common in production).

## Proposals

### 1. Add `__launch_bounds__` to All Kernels (Easy, Immediate)
`__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` hints to the compiler to
cap register usage to ensure the target occupancy:

```cpp
// In energy.h, for GGA density:
template<class scalar_type, bool lda>
__launch_bounds__(64, 4)  // max 64 threads, at least 4 blocks/SM → ≥25% occupancy
__global__ void gpu_compute_density(...) { ... }
```

If registers exceed the budget for the `minBlocksPerSM` target, the compiler spills
to local memory — verify with `--ptxas-options=-v` that spills are minimal.

For `gpu_compute_functions`:
```cpp
__launch_bounds__(128, 2)  // 128 threads, 2 blocks/SM → 2×128=256 threads = 4 warps
```

### 2. Profile-Guided Block Size Selection

Add a one-time `cudaOccupancyMaxPotentialBlockSize` call per kernel during
initialization to select the optimal block size:
```cpp
// In PointGroupGPU::init() or a static init function:
int optBlockSize, minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optBlockSize,
    gpu_compute_functions<scalar_type>, 0, 0);
// Store optBlockSize in PointGroupGPU and use in kernel launch
```

This dynamically selects the block size that maximizes occupancy for the current
GPU and kernel register usage.

### 3. Shared Memory vs L1 Configuration (sm_60+ feature)
SM 6.1 allows configuring the L1/shared split. For kernels with small shared memory
usage (after warp shuffle optimization), prefer more L1:
```cpp
cudaFuncSetCacheConfig(gpu_compute_density<float,false>, cudaFuncCachePreferL1);
```
After `optimize_warp_shuffle.md` reduces shared usage to ~96 bytes, maximizing L1
helps cache the RMM texture reads.

### 4. Tune `DENSITY_BLOCK_SIZE` for GGA Mode
The GGA density kernel uses 4 vec3 shared arrays of size 64 floats = 3 KB. After
warp-shuffle replacement, shared drops to ~96 bytes. This may allow increasing
`DENSITY_BLOCK_SIZE` from 64 to 128, processing more functions per launch and
improving arithmetic density per kernel invocation.

### 5. Profile `gpu_compute_weights` — Potential for 2×
`gpu_compute_weights` loads all `gpu_atoms` (global constant) atom positions into
shared memory. For systems with many atoms, this dominates. Current block size 128
with ~45 regs/thread = 36 concurrent warps (57% occupancy). Reducing shared memory
or splitting the atom-loop could help, but the algorithm is O(N²) — see
`optimize_weight_cache.md` for a more structural improvement.

## Tools to Use

```bash
# See register/shared usage for all kernels:
nvcc -gencode arch=compute_61,code=sm_61 \
     --ptxas-options=-v \
     -I../.. g2g/cuda/iteration.cu 2>&1 | grep "ptxas info"

# Or use nvprof/ncu for runtime occupancy:
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --target-processes all ./liosolo ...
```

## Impact
- `__launch_bounds__` on GGA density: **10–25%** speedup if register spilling is
  acceptable (verify with `-v` flag).
- Dynamic block size selection: **5–15%** for smaller groups where default block size
  overshoots.
- L1 preference: **5–10%** for memory-bound kernels.
- Combined: **15–30% improvement** for GGA DFT runs.

## Difficulty Assessment
**Low–Medium**

- `__launch_bounds__`: trivial 1-line change per kernel, test carefully for spills.
- `cudaOccupancyMaxPotentialBlockSize`: requires storing per-kernel optimal block size
  at startup; medium refactor.
- `cudaFuncSetCacheConfig`: 1 call per kernel, added to `PointGroupGPU::initialize()`.

Files: `g2g/common.h`, `g2g/cuda/kernels/energy.h`, `g2g/cuda/kernels/functions.h`,
`g2g/cuda/kernels/weight.h`, `g2g/cuda/kernels/energy_derivs.h`,
`g2g/cuda/iteration.cu`.

## Estimations
- Total end-to-end improvement: **10–20%** for GGA runs.
- LDA runs: smaller benefit (~5%) since LDA density is already at 67% occupancy.
- Implementation: 1 day for `__launch_bounds__` + profiling; 3 days for dynamic tuning.
