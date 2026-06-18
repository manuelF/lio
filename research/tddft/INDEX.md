# research/tddft — RT-TDDFT / Ehrenfest dynamics

Optimization research for the time-dependent propagation path (driven from
`lioamber/TD.f90` + `lioamber/propagators.f90`). Benchmark workload:
`07_TDDFTHCL/chloride.in` (open shell, propagator=2). **Always benchmark TD on
the long input (ntdstep=50000), never the short one** — per-step overheads are
invisible at a few hundred steps.

| File | Status | Impact | Summary |
|---|---|---|---|
| [xc_launch_overhead_group_merge_2026_06_16](xc_launch_overhead_group_merge_2026_06_16.md) | DONE | high | Predictor XC Fock build is host-launch-bound (GPU 18% util, 58 launches/step). TD-only single-cube grid merge + GlobalMemoryPool re-init fix → **-18.4% wall**. Not bit-exact (FP32 reorder). |
| [td_merge_cpu_disable_2026_06_18](td_merge_cpu_disable_2026_06_18.md) | DONE | high (CPU) | The single-cube merge above is GPU-only: on a CPU build the merged group runs serially on 1 core (`Partition::solve` gives each group `inner_threads==1`). Gated the merge on `gpu_threads>0` → CPU keeps the multi-group partition. **chloride.in 50000 steps 380→128.5 s (−66%)**, dipole test PASS (matches pre-merge `.ok`). SCF untouched (never merges). Remaining CPU lever: within-group point parallelism (risky, shared w/ SCF) or finer TD-CPU cube size. |
| [inter_step_gap_2026_05_21](inter_step_gap_2026_05_21.md) | PARTIAL | medium | Inter-step GPU idle pool; magnus host-buffer caching landed (-2.9%). Vector 4 (BLAS thread crank) DEAD; vector 1 (GPU propagator) dead for small M. |
| [multi_stream_gpu_workers_2026_05_21](multi_stream_gpu_workers_2026_05_21.md) | REJECTED | — | `LIO_GPU_THREADS=2` is 3× slower on TDDFT (host driver/cuBLAS/OMP contention). Kept opt-in, default 1. |
