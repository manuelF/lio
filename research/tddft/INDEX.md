# research/tddft — RT-TDDFT / Ehrenfest dynamics

Optimization research for the time-dependent propagation path (driven from
`lioamber/TD.f90` + `lioamber/propagators.f90`). Benchmark workload:
`07_TDDFTHCL/chloride.in` (open shell, propagator=2). **Always benchmark TD on
the long input (ntdstep=50000), never the short one** — per-step overheads are
invisible at a few hundred steps.

| File | Status | Impact | Summary |
|---|---|---|---|
| [xc_launch_overhead_group_merge_2026_06_16](xc_launch_overhead_group_merge_2026_06_16.md) | DONE | high | Predictor XC Fock build is host-launch-bound (GPU 18% util, 58 launches/step). TD-only single-cube grid merge + GlobalMemoryPool re-init fix → **-18.4% wall**. Not bit-exact (FP32 reorder). |
| [inter_step_gap_2026_05_21](inter_step_gap_2026_05_21.md) | PARTIAL | medium | Inter-step GPU idle pool; magnus host-buffer caching landed (-2.9%). Vector 4 (BLAS thread crank) DEAD; vector 1 (GPU propagator) dead for small M. |
| [multi_stream_gpu_workers_2026_05_21](multi_stream_gpu_workers_2026_05_21.md) | REJECTED | — | `LIO_GPU_THREADS=2` is 3× slower on TDDFT (host driver/cuBLAS/OMP contention). Kept opt-in, default 1. |
