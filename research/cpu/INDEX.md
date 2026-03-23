# CPU Kernel Optimizations

Optimizations for the CPU code path (`g2g/cpu/`) and OpenMP threading.
CPU threads handle small point groups (< 200 points) and currently finish
well within the GPU thread's time, so these are lower priority than GPU work.

## Files

| File | Impact | Summary |
|------|--------|---------|
| [optimize_cpu_threading.md](optimize_cpu_threading.md) | MEDIUM | Replace static bin-packing with OpenMP tasks for dynamic load balancing |
| [optimize_cpu_blas.md](optimize_cpu_blas.md) | MEDIUM | Replace manual density/RMM loops with DGEMM/SYRK |
| [optimize_cpu_memory.md](optimize_cpu_memory.md) | LOW | Use BLAS transpose flags; aligned memory allocation |
| [optimize_cpu_vectorization.md](optimize_cpu_vectorization.md) | LOW | SIMD intrinsics for basis function evaluation loop |
| [optimize_cpu_screening.md](optimize_cpu_screening.md) | LOW | Block screening for CPU basis functions |
