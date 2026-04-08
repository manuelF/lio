# Optimization: Async Execution and CPU-GPU Overlap

**Status:** CLOSED — Phases 1-2 implemented; remaining phases have negligible ROI (0.2% wall)
**Last updated:** 2026-04-08

## What Was Done

### Phase 1 — Pinned Memory (DONE)
Pinned host memory for all PointGroupGPU transfer buffers. Result:
cudaMemcpyAsync 1636ms → 7.5ms (218×), −2.2% wall time.

### Phase 2 — GPU RMM Gather/Scatter (DONE)
`gpu_gather_rmm` + `gpu_scatter_rmm` replace CPU `get_rmm_input()` / `add_rmm_output()`.
Eliminates per-group `cudaStreamSynchronize` for RMM during the SCF loop.
cudaStreamSynchronize calls: 2030 → 151. Actual speedup: **~5% wall** (not the
originally estimated 20–40%, because `fgm=-1` caching had already reduced the number
of groups needing gather/scatter).

## What Remains — And Why It's Not Worth Doing

### Phase 3 — GPU-side Force/Energy Accumulation (NOT RECOMMENDED)

**Originally estimated:** 20–40% speedup
**Actual expected gain:** ~0.2% (~5.5ms on a 3.22s run)

#### Detailed sync call accounting (fosfatoQMMM, 42 GPU groups, 25 SCF iters)

The 151 remaining `cudaStreamSynchronize` calls (837ms total) break down as:

| Source | Location | When called | Calls | Approx time |
|--------|----------|-------------|-------|-------------|
| Fock download | `partition.cpp:620` | Every SCF iter (`compute_rmm=true`) | 25 | ~675ms |
| Energy readback | `iteration.cu:435` | Post-SCF energy-only call | 42 | ~22ms |
| Energy readback | `iteration.cu:435` | Post-SCF forces call | 42 | ~22ms |
| Forces readback | `iteration.cu:546` | Post-SCF forces call | 42 | ~111ms |
| **Total** | | | **151** | **~830ms** |

#### Why the original estimate was wrong

The 20–40% estimate was written **before** Phases 1-2 were implemented. At that time,
the dominant cost was per-group RMM sync during the SCF hot loop (2000+ calls, 25 iters).
After Phase 2, the SCF loop has **zero per-group syncs** — only one structural Fock
download sync per iteration.

The remaining 126 post-SCF syncs (energy + forces) are **not in the hot loop**. They run
once after SCF convergence. The "pipeline gap" each sync introduces is:
- Forces CPU scatter: ~50µs/group (34 atoms × ~1.5µs)
- Energy CPU sum: ~5µs/group
- Sync driver overhead: ~10µs/call
- **Total per-group gap: ~65µs × 42 groups × 2 passes ≈ 5.5ms**

On a 3.22s run: **0.17%**. Well within measurement noise.

#### The 837ms is NOT overhead — it's GPU execution time

The `cudaStreamSynchronize` time is where the CPU thread blocks while the GPU
does real work. The 675ms in the 25 Fock download syncs IS the GPU computing density
+ RMM across all 42 groups per iteration (~30ms kernel time × 25 iters, minus CPU
launch overlap). This time cannot be reduced by eliminating syncs — the GPU kernels
take the same time regardless of how we synchronize.

#### MD simulations don't change the calculus

In molecular dynamics, forces are computed once per MD step (after SCF converges).
The forces sync overhead is ~2.7ms per step (42 syncs × 65µs gap) — still 0.08%
of a ~3.2s MD step. Not significant even over 1000 steps.

#### Larger systems scale proportionally

For N=500 basis, ~500 GPU groups: 500 syncs × 65µs = ~32ms. But GPU kernel time also
scales to ~500ms+. The overhead ratio stays at ~6%, but this is the upper bound and
only applies to the post-SCF forces pass (run once, not 25×).

### Phase 4 — Double Buffering Pipeline (NOT RECOMMENDED)

**Originally estimated:** 10–20% additional
**Actual expected gain:** Near zero

With `fgm=-1` caching, `compute_functions` and transpose run only on iteration 1.
The remaining per-group work is: gather → density → accumulate → rmm → scatter, all
on stream 0. Double buffering would overlap CPU launch prep with GPU kernel execution,
but the CPU launch overhead (~12µs × 5 kernels = 60µs/group) is already negligible
compared to kernel execution (~700µs/group). The GPU is never starved for work.

### Phase 5 — Timer Bypass in Production (LOW PRIORITY)

Still valid but low impact: `cudaEventRecord` calls add ~5-10µs per group per timer.
For 42 groups × ~10 timers × 25 iters = ~525µs total. Trivial.

## Current Bottleneck Breakdown (fosfatoQMMM, 3.22s wall)

| Bottleneck | Time | % Wall | Actionable? |
|------------|------|--------|-------------|
| Fortran SCF overhead (converger, int3lu, DIIS) | ~1.1s | 34% | CPU-GPU overlap (arch change) |
| GPU kernel execution (density+rmm, 25 SCF iters) | ~750ms | 23% | Kernel optimization (roofline) |
| Fock download sync (structural, 25×) | ~675ms | 21% | Cannot eliminate — need Fock for next iter |
| AINT post-SCF (one-time) | ~418ms | 13% | Separate optimization target |
| CPU barrier idle (15 threads × 10ms × 25 iters) | ~250ms | 8% | Speed up GPU or overlap Fortran work |
| Post-SCF syncs (energy + forces) | ~5.5ms | 0.2% | GPU-side scatter (not worth it) |

## What Would Actually Help (ranked)

1. **CPU-GPU Fock overlap** — Run int3lu (CPU Coulomb, ~11% of CPU time) concurrently
   with g2g GPU work. Currently the Fortran SCF loop is strictly sequential:
   `g2g_solve_groups → int3lu → converger → next iter`. If Fock buffers were separated
   (XC vs Coulomb), int3lu could run during the GPU's density+rmm phase, recovering
   most of the 250ms barrier idle. **Difficulty: High** (Fortran architectural change).

2. **GPU kernel optimization** — The density kernel is 45% of GPU time and has room
   on the roofline. See `roofline_gpu_compute_density.md`. Even a 10% improvement in
   the density kernel saves ~60ms wall time — more than GPU-side forces scatter would.

3. **Open-shell register pressure** — 93 → 56 registers for open-shell GGA would
   improve occupancy from 34% → 56%. Irrelevant for closed-shell fosfatoQMMM but
   significant for open-shell benchmarks. See `optimize_open_shell_registers.md`.

## Historical Context

This file originally proposed 4 phases with combined 35–60% speedup estimates.
Those estimates were based on the pre-optimization state (2030 sync calls, no RMM
caching). After implementing Phases 1-2 plus `fgm=-1` caching, the landscape changed
fundamentally:

- **Before:** Per-group syncs dominated SCF iteration time (2030 calls/25 iters = ~81/iter)
- **After:** SCF iterations are sync-free except one structural Fock download per iter
- **Remaining syncs:** 126 post-SCF calls with ~5.5ms total pipeline overhead

The lesson: **always re-profile after each optimization**. Removing one bottleneck
can shift the cost structure so dramatically that previously high-impact items become
irrelevant.
