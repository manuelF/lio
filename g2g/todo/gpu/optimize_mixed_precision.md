# Optimization: Mixed Precision — Corrected for SM 6.1 (Pascal)

## ⚠ Previous Version Was Wrong for This Hardware

The prior version assumed Tensor Core availability (Volta+ / SM 7.0+).
**GTX 1080 is SM 6.1 (Pascal): there are no Tensor Cores.**

FP16 `__half2` paired-lane MADs exist on Pascal but at ~1/64 of INT8 throughput —
nowhere near the "2x–8x" claim that applies only to Volta/Ampere Tensor Cores.
TF32 and BF16 are Ampere-only (SM 8.0+). This document now reflects SM 6.1 reality.

## What SM 6.1 Actually Provides

| Feature | SM 6.1 Support | Throughput vs FP32 |
|---|---|---|
| FP32 SIMT | ✓ | 1× baseline |
| FP64 SIMT | ✓ | 1/32× — avoid |
| `__half2` MAD | ✓ (CUDA_ARCH≥530) | ~0.02× — nearly useless |
| `__ldg()` L1 cache | ✓ | Bandwidth, not compute |
| Tensor Cores (wmma) | ✗ | Volta+ only |
| TF32 mode | ✗ | Ampere+ only |
| BF16 | ✗ | Ampere+ only |
| cuBLAS `CUBLAS_TF32_TENSOR_OP_MATH` | ✗ | Ampere+ only |

## Realistic Proposals for SM 6.1

### 1. FP16 Storage for `function_values` — Bandwidth Reduction (Best ROI)
Store `gpu_compute_functions` output as `__half` instead of `float`. This halves
the bandwidth consumed by the largest intermediate array.

- `function_values`: M×P floats. Typical group: M~30, P~512 → ~60 KiB → 30 KiB as `__half`.
  SM 6.1 L2 is 1.5 MiB, so fitting both float/half copies is feasible.
- Arithmetic in `gpu_compute_density` and `gpu_update_rmm` still uses FP32 after
  converting: `float fj = __half2float(fv_half[idx]);`
- Risk: FP16 underflows below ~6×10⁻⁵. GTO exp(−α·r²) values near the `exp>70` cutoff
  are already zeroed, so underflow risk is low. Validate against `agua` and `fosfato`.

**Estimated impact: 5–15% end-to-end speedup** (L2 hit rate and bandwidth savings).

### 2. Reduce FP64 Accumulation in iteration.cu (Clean, Easy)
Some accumulations in `iteration.cu` use `double` (e.g., `local_energy += ...`).
Reducing these to `float` where precision allows cuts register use and enables
more warps to be active simultaneously (warp occupancy boost).

This is safe for energies where SCF convergence tolerances are ~10⁻⁶ Hartree.
For forces (geometry optimization / MD), keep double accumulation.

**Estimated impact: 3–8% for energy-only runs.**

### 3. DP4A INT8 for Far-Field Screening (Low Priority)
Pascal has DP4A (INT8 dot product) for neural networks. Not directly applicable
to GTO basis function evaluation. Skip.

## Future Hardware Upgrade Path (Ampere+ / SM 8.0+)
If the GPU is upgraded, immediately revisit:
- `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` — free ~2× on SYRK/GEMM
- `wmma::` Tensor Core tiles for the density GEMM reformulation
- BF16 storage (wider dynamic range than FP16, safer for GTOs)

## Priority Assessment
**Low** for current GTX 1080 hardware. Pursue warp-shuffle reductions
(`optimize_warp_shuffle.md`) and the RMM gather-on-GPU (`optimize_rmm_gather_gpu.md`)
first — both give larger, hardware-appropriate gains.

## Difficulty Assessment
**Low–Medium** for FP16 storage only.

Files: `g2g/cuda/kernels/functions.h` (write `__half`), `g2g/cuda/kernels/energy.h`
and `rmm.h` (read `__half`, accumulate `float`), `g2g/matrix.h` (optional
`CudaMatrix<__half>` specialization or just cast raw `uint16_t*` pointer).

## Estimations
- FP16 function_values storage: **5–15% overall speedup** (bandwidth, L2 hit rate).
- FP32 accumulator reduction: **3–8%** for energy-heavy runs.
- Tensor Core path: **N/A on GTX 1080; 2–5× on A100/H100 with cuBLAS GEMM**.
