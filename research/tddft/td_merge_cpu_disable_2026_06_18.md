# TD single-cube grid merge is counterproductive on CPU — DONE 2026-06-18

**Status:** DONE. Shipped. 50000-step dipole test PASS (matches pre-merge `.ok`).

## TL;DR

The TD-only single-cube grid merge ([[xc_launch_overhead_group_merge_2026_06_16]])
collapses the whole XC grid into one cube `PointGroup` for small systems (M≤80).
On the **GPU** that's a launch-reduction win (one big launch instead of dozens of
tiny ones). On a **CPU-only build** it backfires badly: `Partition::solve` hands
each group to a single thread with `inner_threads==1`, so one merged group means
the entire XC solve runs **serially on one of N cores**.

Gated the merge on `G2G::gpu_threads > 0` (one line in
`regenerate_partition.cpp`), so CPU builds keep the normal multi-group partition
and spread groups across CPU threads.

| chloride.in, 50000 steps, 8-core CPU | merge on (before) | merge off (after) |
|---|---|---|
| `TD - Pred XC (g2g)` | ~357 s | **120 s** |
| **total wall** | **~380 s** | **128.5 s (−66%)** |

(Same −66% at 5000 steps: 38.0 → 13.4 s.) The predictor XC Fock build is 94–98%
of TD wall, so this is essentially the whole runtime.

## Why it serializes on CPU

`g2g_solve_groups` → `Partition::solve` runs
`#pragma omp parallel for num_threads(cpu_threads)` over **work bins**, each bin a
list of groups solved serially with `inner_threads==1`. Parallelism therefore
comes from *having multiple groups*. One merged group ⇒ one bin busy, the rest
idle. The within-group loops (functional eval over points, `cpu_update_rmm` Fock
accumulate) are not parallelized (inner_threads==1).

Confirmed by timing: merge-on TD XC = 7.1 ms/step (1 core); merge-off = 2.4 ms/step
(~3× — the unmerged HCl partition is ~1 big cube + 2 atom spheres ≈ 3 groups).

## Correctness

The merge was never bit-exact (FP32 grid-reduction reorder); `dipole_moment_td.ok`
predates the merge commits (last touched by PR #281/#283; merge landed in
`d6c00c0a`/`5a7bd661`, which did **not** regenerate it). Disabling the merge on
CPU restores the pre-merge grouping, so the 50000-step dipole now matches `.ok`
within the 1e-3 test tolerance — i.e. this *fixes* the CPU test, which the
merge-on path had been perturbing.

## Scope / safety

Gate is `td_merge_groups && M≤80 && gpu_threads>0`. `td_merge_groups` is only set
during the TD partition rebuild (`TD.f90` `td_integration_setup`), and SCF never
merges, so SCF / heme FP-summation order is completely untouched. GPU builds
(`gpu_threads>0`) keep the merge and its −18.4% GPU win.

## Remaining TD-CPU lever (not done)

~3× is bounded by the ~3-group partition; the big cube group still runs on one
core. Full N× would need **within-group point parallelism** (inner_threads>1) in
`Partition::solve`, but that path is shared with SCF and FP-order-sensitive
(heme), so it was left untouched. A TD-CPU-only finer cube size (more groups) is
the lower-risk follow-up if more is needed.
