# OpenMP Parallelism Analysis in g2g SCF Loop

**Status:** DONE — analysis complete, dead pragmas removed
**Impact:** Minor cleanup (~18-53ms saved), key insight: barrier idle is GPU-bound
**Date:** 2026-03-28

## Architecture

The g2g SCF iteration uses a single `#pragma omp parallel for` in `partition.cpp:556`:
- `cpu_threads + gpu_threads` total (typically 15 + 1 = 16)
- `schedule(static)` — each thread gets one pre-packed work bin
- Threads 0..14 process CPU PointGroups, thread 15 processes GPU PointGroups
- Implicit barrier at loop end synchronizes all threads

## Measured Balance (fosfatoQMMM, M=86, 25 SCF iters)

```
CPU max: ~22ms (710 groups / 15 threads, ~47 groups/thread)
GPU:     ~32ms (42 groups / 1 thread)
Idle:    ~10ms per iteration (CPU waits for GPU at barrier)
Total idle: 250ms wall, 3.75s CPU-core time over 25 iterations
```

## Root Cause of 49% CPU Barrier Idle (perf)

The 49% of CPU time in `gomp_team_barrier_wait_end` is NOT caused by:
- Over-parallelization or too many parallel regions
- Nested OMP overhead (was present but negligible)
- Poor load balancing between CPU threads (rebalancer works well)

It IS caused by:
- **GPU thread is the critical path** (32ms vs CPU 22ms)
- 15 CPU threads finish, then wait ~10ms for the 1 GPU thread
- GPU time dominated by kernel execution (density 602ms + rmm 191ms / 25 = ~32ms/iter)
- Forces cudaStreamSynchronize adds ~5ms/iter to GPU thread time

## What Was Changed

Removed 10 dead `#pragma omp parallel for num_threads(inner_threads)` directives from
`g2g/cpu/iteration.cpp`. These had `inner_threads=1` (hardcoded), making them serial
loops with OMP runtime fork/join overhead (~1-3µs each × 710 groups × 25 iters).

**Impact:** Negligible on wall time (within noise), but cleaner code and slightly
reduced CPU user time.

## What Would Actually Help

1. **GPU-side force accumulation** (research/gpu/async_execution.md priority #1):
   Eliminate 840ms of cudaStreamSynchronize. Would reduce GPU thread from ~32ms to ~24ms,
   nearly matching CPU (22ms) and eliminating most barrier idle.

2. **CPU-GPU Fock overlap** (separate Fortran-level change): Run int3lu (CPU Coulomb)
   concurrently with g2g_solve_groups (GPU XC). Uses idle CPU time productively.

## Thread Layout Reference

```
Thread 0..14 (CPU):  process work[0..14] → PointGroupCPU::solve_closed()
Thread 15 (GPU):     process work[15]    → PointGroupGPU::solve_closed()
                     cudaStreamSynchronize(0) at end
                     download_gpu_fock()

[IMPLICIT BARRIER — CPU threads idle here]

Serial post-loop: Kahan accumulation of forces + RMM (negligible: ~100-200µs)
Rebalancer: adjusts work[] bins for next iteration (~100µs)
```

## OMP Directives Inventory (after cleanup)

| File | Line | Directive | Per-iteration? |
|------|------|-----------|----------------|
| partition.cpp:444 | `parallel for schedule(guided,8)` | No (once at grid gen) |
| partition.cpp:449 | `parallel for schedule(guided,8)` | No (once at grid gen) |
| partition.cpp:556 | `parallel for num_threads(N) schedule(static)` | **Yes (1/iter)** |
| cpu/weight.cpp:34 | `parallel for` | No (once at grid gen) |
| cpu/functions.cpp:50 | `parallel for schedule(static)` | No (once at grid gen) |
