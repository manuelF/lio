# SCF Loop Profiling: fosfatoQMMM (2026-03-28)

**Status:** DONE — baseline profile post allocation-hoisting optimizations
**Impact:** Informational (guides next optimization)
**System:** fosfatoQMMM, 34 QM atoms, M=86, closed-shell GGA, GTX 1080 SM 6.1

## Timing Summary

- **Wall time:** 4.09-4.14s (5 runs)
- **User time:** 25.4-26.5s (multi-threaded, 15 OpenMP threads)
- **SCF iterations:** 25
- **Per-iteration:** ~130ms wall

## GPU Kernel Breakdown (nsys)

| Kernel | Time (ms) | % GPU | Instances | Avg (ms) |
|--------|-----------|-------|-----------|----------|
| gpu_compute_density | 602 | 51.7% | 1,134 | 0.53 |
| gpu_update_rmm | 191 | 16.4% | 1,050 | 0.18 |
| gpu_compute_density_derivs | 75 | 6.5% | 42 | 1.79 |
| AINT forces (all) | 205 | ~18% | 12 | varies |
| AINT fock (all) | 30 | ~2.6% | 12 | varies |
| transpose | 7 | 0.6% | 84+42 | 0.08 |
| gpu_compute_functions | 4 | 0.3% | 42 | 0.09 |
| gpu_gather/scatter_rmm | 6 | 0.5% | 2,184 | 0.003 |

**Total GPU kernel time: ~1.16s**

## CUDA API Bottleneck

| API Call | Time (ms) | % | Calls |
|----------|-----------|---|-------|
| cudaStreamSynchronize | 840 | 63.8% | 151 |
| cudaDeviceSynchronize | 214 | 16.3% | 6 |
| cudaLaunchKernel | 74 | 5.6% | 5,836 |
| cudaFree | 66 | 5.0% | 1,239 |
| cudaMemcpy | 40 | 3.1% | 624 |
| cudaMalloc | 29 | 2.2% | 1,240 |

**Key finding:** 840ms in cudaStreamSynchronize (151 calls) = 63.8% of API time.
151 calls = ~3 per group × ~42 groups + overhead. These are per-group energy+forces syncs.

## CPU Profile (perf)

| Function | % CPU |
|----------|-------|
| gomp_team_barrier_wait_end | 25.1% |
| gomp_barrier_wait_end | 23.7% |
| cpu_compute_density_gga<float> | 10.4% |
| dgemv_kernel (int3lu) | 7.6% |
| dgemm_kernel (converger) | 3.8% |
| sgemv_kernel (int3lu) | 3.4% |
| compute_optimal_split_cost | 1.3% |

**Key finding:** ~49% of CPU time is idle (OpenMP barrier waits).
CPU threads wait for GPU threads to finish their work bins.

## Memory Operations

- Total memcpy time: ~12ms (negligible)
- cudaMalloc+cudaFree: ~95ms (1,240 allocs + 1,239 frees)
- Still 1,240 malloc/free pairs despite fgm=-1 caching (these are non-function buffers)

## Conclusions

1. **cudaStreamSynchronize is the #1 bottleneck (840ms/4.1s = 20.5% wall time)**
   - Confirms research/gpu/async_execution.md priority #1
   - Fix: GPU-side force accumulation (like existing GPU-side Fock scatter)

2. **CPU is 49% idle** — massive overlap opportunity
   - int3lu (CPU Coulomb) uses ~11% of CPU time
   - Converger DGEMMs use ~4% of CPU time
   - Both could run while GPU works if Fock buffers are separated

3. **cudaMalloc/cudaFree (95ms)** — still significant
   - 1,240 pairs suggest non-function buffers aren't cached
   - Worth investigating which allocations bypass the pool

4. **Allocation hoisting (this change)** — no measurable wall-time impact at M=86
   - Expected: heap alloc is ~1us each, 50 allocs/iter × 25 iters = ~1.25ms
   - Value is in code cleanliness and reduced fragmentation for larger systems
