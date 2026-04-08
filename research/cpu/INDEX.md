# CPU Kernel Optimizations

Optimizations for the CPU code path (`g2g/cpu/`) and OpenMP threading.
CPU threads handle 304 small groups in ~13ms/iter (fosfatoQMMM, RTX 3080 Ti),
well within the GPU thread's ~18ms. These are low priority — the CPU/GPU
balance is good (4ms idle/iter).

**Note (2026-04-08):** The much larger Fortran-side CPU bottleneck (converger,
int3lu = 35% of wall) is tracked in `research/fortran/`, not here.

## Files

| File | Impact | Summary |
|------|--------|---------|
| [optimize_cpu_threading.md](optimize_cpu_threading.md) | LOW (was MEDIUM) | CPU already balanced at 13ms vs 18ms GPU; only 4ms idle/iter |
| [optimize_cpu_blas.md](optimize_cpu_blas.md) | LOW | Replace manual density/RMM loops with DGEMM/SYRK; g2g CPU is only part of 11% |
| [optimize_cpu_memory.md](optimize_cpu_memory.md) | LOW | Use BLAS transpose flags; aligned memory allocation |
| [optimize_cpu_vectorization.md](optimize_cpu_vectorization.md) | LOW | SIMD intrinsics for basis function evaluation loop |
| [optimize_cpu_screening.md](optimize_cpu_screening.md) | LOW | Block screening for CPU basis functions |
