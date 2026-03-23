# Optimization: GPU-side RMM Gather (Eliminate get_rmm_input CPU Bottleneck)

## Summary

**This is the #1 priority optimization for GPU throughput** (also listed as #1 in
`upcoming_utilization.txt`).

Every call to `solve_closed` and `solve_opened` in `iteration.cu` must first call
`get_rmm_input()` on the CPU. This function gathers a group-local submatrix from the
global Fortran density matrix (RMM) — applying index remapping, lower-triangle packing,
and optional sqrt(2) symmetry factors — then copies it to GPU via `cudaMemcpy2DToArrayAsync`.

This creates a mandatory CPU serialization point: the GPU sits idle while the CPU
performs the gather loop. For large systems with many groups, this overhead is dominant.

## Code Audit: get_rmm_input

`partition.cpp` (via `PointGroupCPU` and `PointGroupGPU` overrides):
```cpp
void PointGroupGPU::get_rmm_input(HostMatrix<scalar_type>& rmm_input) const {
  rmm_input.resize(group_m, group_m);  // Always reallocates
  for (int i = 0; i < group_m; i++) {
    for (int j = 0; j <= i; j++) {
      uint global_i = local2global_func[rmm_rows[i]];  // Indirect index
      uint global_j = local2global_func[rmm_cols[j]];
      rmm_input(i, j) = (scalar_type)fortran_rmm(global_i, global_j);
    }
  }
}
```

The double-loop is O(M²) with indirect accesses — poor cache behavior. Then
`cudaMemcpy2DToArrayAsync` stages this into a CUDA array for texture.

The Fortran RMM `fortran_rmm` is already in CPU RAM (FortranMatrix, column-major).
It is NOT on the GPU, which is the fundamental problem.

## Proposal: Keep a GPU Copy of the Global RMM

### Step 1 — Upload Global RMM Once Per SCF Step

At the start of each `Partition::solve` call (once, not per group), upload the full
global RMM to GPU:
```cpp
// In Partition::solve (partition.cpp):
if (compute_rmm || use_density) {
  int N = fortran_vars.m;  // Total basis functions
  rmm_global_gpu.resize(N, N);
  cudaMemcpy(rmm_global_gpu.data, fortran_vars.rmm.data,
             N * N * sizeof(double), cudaMemcpyHostToDevice);
}
```
This is one large transfer per SCF step vs one small-but-CPU-prepared transfer per group.
For N=500: 500×500×8 = 2 MB at ~10 GB/s PCIe → ~0.2 ms total. Per group with M=30:
30×30×4 = 3.6 KB → was ~0.036 ms/group × 100 groups = 3.6 ms. Net saving for 100 groups:
**3.4 ms/SCF step** plus elimination of the CPU loop time.

### Step 2 — Replace get_rmm_input with a GPU Gather Kernel

```cpp
// New kernel: g2g/cuda/kernels/rmm_gather.h
__global__ void gpu_gather_rmm(
    const double* __restrict__ global_rmm,  // N×N global (col-major Fortran)
    const uint*   rows,       // group_m entries: local→global row map
    const uint*   cols,       // group_m entries: local→global col map
    float*        local_rmm,  // group_m×group_m output (lower triangle)
    int N, int group_m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= group_m || j > i) return;
  uint gi = rows[i];  // global index for local function i
  uint gj = cols[j];  // global index for local function j
  // Fortran col-major: rmm[gi][gj] = global_rmm[gj * N + gi]
  local_rmm[i * group_m + j] = (float)global_rmm[gj * N + gi];
}
```

Launch: `dim3(divUp(group_m, 16), divUp(group_m, 16))` with `dim3(16, 16)`.
For M=30: 2×2 blocks = 4 blocks, trivially fast.

`rows` and `cols` are the existing `PointGroup::rmm_rows` and `rmm_cols` (see
`partition.h`). They're small (M entries each) and can be uploaded once per group
to constant memory or GPU buffer.

### Step 3 — Feed Gathered RMM Directly as Linear Buffer (eliminates texture)
If combined with `optimize_density_texture.md` Part B, the gathered `local_rmm`
can be used directly with `__ldg` instead of staging into a CUDA array. This avoids
the `cudaMemcpy2DToArray` call entirely.

### Step 4 — GPU-side RMM Scatter (add_rmm_output on GPU)

After `gpu_update_rmm` computes the new local RMM, the current code copies back to
CPU and runs `add_rmm_output`:
```cpp
// Current: D2H + CPU scatter
rmm_output_host.copy_submatrix_async(rmm_output_gpu, 0);
cudaDeviceSynchronize();
this->add_rmm_output(rmm_output_host, ...);  // CPU scatter
```

Replace with a GPU scatter kernel:
```cpp
__global__ void gpu_scatter_rmm(
    const float* local_rmm, const uint* rows, const uint* cols,
    double* global_rmm, int N, int group_m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= group_m || j > i) return;
  uint gi = rows[i]; uint gj = cols[j];
  atomicAdd(&global_rmm[gj * N + gi], (double)local_rmm[i * group_m + j]);
}
```
Note: `atomicAdd` on `double` requires SM 6.0+ (Pascal supports it natively).
GTX 1080 (SM 6.1) has hardware double atomics. ✓

## Impact

| Step | Saves | Complexity |
|---|---|---|
| Global RMM upload (Step 1) | Eliminates per-group CPU gather loop | Low |
| GPU gather kernel (Step 2) | Eliminates per-group H2D copy + CPU work | Medium |
| GPU scatter kernel (Step 4) | Eliminates per-group D2H + CPU scatter | Medium |
| Texture elimination (Step 3) | Removes staging overhead | Low (if Step 2 done) |

**Combined effect: 20–40% overall runtime improvement** for large systems.
For small systems (few groups, large M), the per-group benefit is smaller but
the reduced latency improves interactive throughput.

## Correctness Risks

**High — this is the most correctness-sensitive optimization:**
1. Fortran column-major indexing: `rmm[gi][gj]` = `data[gj*N + gi]`. Getting this
   wrong produces plausible-but-wrong densities that may converge to the wrong minimum.
2. The `(ii==j ? 2 : 1)` symmetry factor in `get_rmm_input` for the closed-shell case.
   Must be applied correctly in the gather kernel.
3. Open-shell: `get_rmm_input(a, b)` gathers two separate RMM matrices (alpha and
   beta spin). Both need separate gather kernels.
4. Race condition in GPU scatter: multiple groups may scatter to overlapping global_i/gj
   indices if groups share basis functions. This is handled by `atomicAdd` but test
   thoroughly — use `agua` and `Fe3H2O6` energy+forces as regression benchmarks.

**Validation protocol:**
1. Implement Step 1 only first. Compare energies/forces to reference.
2. Add Steps 2+3. Re-validate.
3. Add Step 4. Re-validate.
4. Run all `test/LIO_test/` tests with `make check`.

## Difficulty Assessment
**Medium–High** per step; **High** combined.

Files to modify:
- `g2g/init.h` / `g2g/init.cpp`: Add `CudaMatrix<double> rmm_global_gpu`.
- `g2g/cuda/iteration.cu`: Upload RMM once; call gather/scatter kernels.
- `g2g/cuda/kernels/rmm_gather.h`: New gather kernel.
- `g2g/cuda/kernels/rmm_scatter.h`: New scatter kernel.
- `g2g/partition.h`: Upload `rmm_rows`, `rmm_cols` to GPU buffers.

## Estimations
- Step 1 alone: **5–15% speedup** (eliminates CPU gather loop, keeps H2D copy).
- Steps 1+2: **15–25% speedup** (eliminates per-group H2D latency).
- Steps 1+2+4: **20–40% speedup** (full gather+scatter on GPU).
- For MD simulations: the per-step cost reduction enables larger systems and shorter
  SCF iteration wall time. **Critical path item for DFT-MD throughput.**
