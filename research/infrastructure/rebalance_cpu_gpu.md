# Rebalancer Analysis: Current Behavior, Problems, and Opportunities

**Status:** ANALYSIS — this document replaces a prior speculative proposal with
findings grounded in measured profiling data and code inspection.

---

## How `rebalance()` Actually Works

`Partition::rebalance()` in `g2g/partition.cpp:431` runs after every SCF iteration.
It moves groups between threads of the **same device type** to equalize per-thread
wall time:

```
for device in {CPU, GPU}:
    for 5 rounds:
        find slowest and fastest thread (within device class)
        while imbalance > 2%:
            pick the group from the slowest thread whose time best halves the gap
            move it to the fastest thread
            (for GPU groups: call deallocate() to release cached GPU memory)
```

### Key architectural constraints

1. **Group type is fixed at creation.** `PointGroupCPU` and `PointGroupGPU` are
   distinct C++ classes with different `solve_closed()` / `solve_opened()`
   implementations (virtual dispatch). A group's device affinity is set in
   `regenerate()` based on `is_big_group()` → `points.size() > SPLITPOINTS`.

2. **Virtual dispatch, not thread dispatch.** The `solve()` loop calls
   `group->solve_closed()` which resolves to `PointGroupCPU::solve_closed()`
   or `PointGroupGPU::solve_closed()` regardless of which OMP thread invokes it.
   A `PointGroupGPU` always launches CUDA kernels; a `PointGroupCPU` always runs
   CPU code.

3. **Single GPU is the common case.** `gpu_threads = cudaGetDeviceCount()`,
   typically 1. With gpu_threads=1, the GPU half of rebalance is a no-op
   (largest == smallest, nothing to move). GPU rebalancing only matters for
   multi-GPU setups.

4. **CPU threads = OMP_NUM_THREADS − gpu_threads.** Typically 15 CPU threads on
   a 16-core machine. CPU groups (≤200 points) are bin-packed across these threads.

### Thread layout (fosfatoQMMM, 16 OMP threads)

```
work[0]  … work[14]  → CPU threads (process PointGroupCPU objects)
work[15]              → GPU thread  (processes all PointGroupGPU objects sequentially)
```

---

## Measured Reality: Who is the Bottleneck?

**Profiled on fosfatoQMMM** (34 QM atoms, 25 SCF iterations, GTX 1080):

| Thread class | Groups/iter | Time/iter (fgm=0) | Time/iter (fgm=-1) |
|---|---|---|---|
| GPU thread (1 thread, ~78 groups) | 78 | ~230 ms | ~150 ms |
| CPU threads (15 threads, many small groups) | varies | < 230 ms | < 150 ms |

**The GPU thread IS the critical path.** CPU threads always finish before the
GPU thread because small groups are cheap and well-distributed. The OMP barrier
at the end of the parallel region means all CPU threads idle-wait for the GPU
thread to finish.

**Implication:** Rebalancing CPU threads is polishing the wrong knob. Even
perfect CPU load balance doesn't reduce wall time — the GPU thread determines
it. The only way to reduce wall time is to:
1. Make the GPU thread faster (caching, fewer mallocs, faster kernels), or
2. Offload some GPU work to idle CPU threads (cross-device migration)

---

## Problem 1: Rebalancer Causes Nondeterminism

**Discovered 2026-03-20 during auto-fgm testing.**

With GPU memory caching enabled (`fgm=-1`), the GPU thread finishes faster
(~150ms vs ~230ms per iteration). This changes the wall-clock timing profile
stored in `timeforgroup[]`, which causes `rebalance()` to make different
group-to-thread decisions. Different CPU threads → different indices in
`rmm_outputs[k]` → different loop order in the post-parallel accumulation:

```cpp
// partition.cpp:637 — accumulation order depends on which thread holds which groups
for (uint k = 0; k < rmm_outputs.size(); k++) {
    const double* src = rmm_outputs[k].asArray();
    for (int i = 0; i < elements; i++)
        dst[i] += src[i];  // float64 addition — order matters for last bits
}
```

Different addition order → different last-bit rounding → different RMM matrix →
different DIIS trajectory → different convergence path. Measured variation:
~0.0002 Ha on Fe3H2O6 (run-to-run with identical inputs).

Without caching (`fgm=0.0`), the timing is perfectly reproducible → rebalance
makes identical decisions every run → deterministic accumulation → deterministic
results.

### Fix: Deterministic accumulation order

**Simple fix:** After `rebalance()`, sort each `work[i]` by group index:
```cpp
for (uint i = 0; i < work.size(); i++)
    sort(work[i].begin(), work[i].end());
```
This doesn't fix the nondeterminism of WHICH groups are on which thread, but
ensures groups within a thread are processed in a fixed order. The accumulation
loop iterates over threads 0..N in order, so the contribution from thread k
is always the same set of groups (after convergence of rebalance, which
stabilizes after 1-2 iterations).

**Better fix:** Accumulate per-group results in group-index order rather than
per-thread order. This requires storing outputs per-group (not per-thread),
which increases memory but makes the accumulation completely order-independent
of thread assignment.

**Best fix (but complex):** Use Kahan-compensated accumulation in the
post-parallel merge loop. This makes the result insensitive to addition order
(compensation absorbs the last-bit differences). Since this is a float64 loop
on the CPU, the overhead is negligible.

---

## Problem 2: GPU-Side Rebalancing Destroys Cache

When `rebalance()` moves a GPU group between GPU threads (multi-GPU only), it
calls `deallocate()` on the group (`partition.cpp:482-486`). This destroys
cached function values, textures, and all persistent GPU state — forcing full
recomputation on the next iteration. With auto-fgm caching, this is a severe
performance penalty.

For single-GPU setups (gpu_threads=1), this is not an issue (no GPU
rebalancing occurs). For multi-GPU, the rebalancer should be aware of the
caching cost and only move groups when the time saved from better load balance
exceeds the time lost from cache eviction.

---

## Assessment of Prior Proposal: Cross-Device Work Stealing

The prior version of this document proposed unified CPU/GPU rebalancing with
"shadow price" memory economics and dynamic type promotion/demotion. Here's
why this doesn't hold up:

### Why cross-device migration is impractical

1. **Type conversion is expensive.** Converting `PointGroupGPU` → `PointGroupCPU`
   (or vice versa) requires:
   - Destroying the current object (freeing all GPU/CPU buffers)
   - Creating a new object of the other type
   - Re-running `compute_weights()` (launches kernels / does CPU work)
   - Re-running `compute_functions()` on the new device
   - Re-running `compute_indexes()` to rebuild index arrays

   This is essentially the full cost of `regenerate()` for that group. It would
   only pay off if the group stays on the new device for many iterations — but
   rebalance runs every iteration and might move it back.

2. **CPU threads are not the bottleneck.** The entire motivation for
   CPU→GPU migration is that CPU threads are overloaded. But they're not — they
   finish before the GPU thread. The GPU thread is the bottleneck. Moving more
   work TO the GPU would make it slower.

3. **GPU→CPU migration could theoretically help** (offload small GPU groups to
   idle CPU threads). But:
   - Small groups are already CPU groups (SPLITPOINTS=200 threshold)
   - GPU groups have 200+ points — too expensive for CPU without GPU parallelism
   - The type conversion cost (see above) likely exceeds any savings

4. **The cost model is uncalibrated.** The prior document proposed
   `T_cpu(n) ≈ α n² + β n` and `T_gpu(n) ≈ α n² + β n + γ` without empirical
   basis. The actual cost function in the code is `npts × M × (1 + M) / 2`
   (O(P·M²)), and the real execution time depends on memory caching, texture hit
   rate, and many other factors not captured by a polynomial model.

5. **"Shadow price" memory model is solved by auto-fgm.** The `fgm=-1`
   auto-detection (implemented 2026-03-20) computes optimal cache budgets
   without runtime economics. For large systems, Phase 2 (priority caching by
   cost/byte ratio) handles the case where not everything fits. No runtime
   pricing needed.

---

## Performance Model (Measured 2026-03-20)

Measured per-group execution time for all 165 groups on fosfatoQMMM (34 QM atoms),
running each group exclusively on CPU (LIO_SPLIT_POINTS=99999) and GPU
(LIO_SPLIT_POINTS=1). Second SCF iteration used to avoid cold-start.

### Fitted models

The dominant cost variable is `P × M²` where P = number of integration points
and M = number of overlapping basis functions per group.

**CPU model** (no fixed overhead, all computation):
```
T_cpu(P,M) = α_cpu × P × M²

α_cpu = 1.967e-03 µs per unit P×M²
R²    = 0.985  (fitted on 117 groups with T > 5µs, filtering timer noise)
```

**GPU model** (significant fixed overhead per kernel launch):
```
T_gpu(P,M) = γ + α_gpu × P × M²

γ     = 356 µs  (fixed overhead: kernel launch, memcpy, cudaMalloc/Free)
α_gpu = 2.604e-05 µs per unit P×M²
R²    = 0.926  (fitted on 164 warm groups, skip first cold-start)
```

**Key ratios:**
- GPU slope speedup: `α_cpu / α_gpu = 75.5×` (GPU massively faster per unit work)
- Fixed overhead: GPU pays 356 µs before any computation begins

### Crossover analysis

Setting `T_cpu = T_gpu`:
```
α_cpu × P × M² = γ + α_gpu × P × M²
P × M² = γ / (α_cpu − α_gpu) = 183,549
```

Below `P×M² = 183,549`, CPU is faster (GPU overhead dominates).
Above this, GPU wins (parallel throughput dominates).

**Crossover points for typical M values in this system:**

| M | P_crossover | T_crossover (µs) | Note |
|---|---|---|---|
| 20 | 459 | 361 | near current default SP=200 |
| 40 | 115 | 361 | trivial group |
| 60 | 51 | 361 | trivial group |
| 80 | 29 | 361 | trivial group |
| 100 | 18 | 361 | trivial group |
| 120 | 13 | 361 | trivial group |
| 150 | 8 | 361 | trivial group |

For the typical groups in fosfatoQMMM (M=83..365, P=13..9261), all groups with
P > ~50 are firmly in GPU territory.

### SPLITPOINTS sweep (fosfatoQMMM, wall time)

| SPLITPOINTS | Wall time | CPU max/iter | GPU/iter | Who waits | Idle/iter |
|---|---|---|---|---|---|
| 200 (default) | 5.82 s | 0.9 ms | 134 ms | CPU | 133 ms |
| 2800 (sweet spot) | 5.49 s | 91 ms | 132 ms | CPU | 41 ms |
| 3200 (cliff) | 9.02 s | 253 ms | 132 ms | GPU | 122 ms |

**Interpretation:** At SP=200, CPU threads are 99.3% idle. Raising SP to 2800
moves some medium groups to CPU, reducing idle time from 133→41 ms/iter (−6%
wall time). At SP=3200, one large group (P=9261, M=365, T_cpu=25ms per
M² evaluation) overwhelms a CPU thread → 2× regression.

### Optimal SPLITPOINTS formula

For a group to be faster on GPU:
```
P × M² > γ / (α_cpu − α_gpu) = CROSSOVER
SPLITPOINTS_optimal(M) = CROSSOVER / M²
```

Since SPLITPOINTS is applied to P only (not P×M²), and M varies per group,
a single static SPLITPOINTS is inherently approximate. The benchmark utility
(`g2g/bench/splitpoint_tune.py`) can compute the optimal value for a given
system by measuring actual per-group times and fitting the model.

---

## What Would Actually Help

Ranked by expected impact, considering the measured bottleneck is the GPU thread:

### 1. Fix nondeterminism (LOW effort, HIGH value for correctness)

Sort `work[i]` after rebalance, or use order-independent accumulation. This
unblocks making `fgm=-1` the default (currently `fgm=0.0` for determinism).
See "Problem 1" above.

### 2. SPLITPOINTS auto-tuning (MEDIUM effort, MEDIUM impact)

The current threshold (default 200 points) is static. A group with 201 points
goes to GPU (high fixed overhead); a group with 199 goes to CPU. This cliff
should be smoothed.

**Approach:** Run the benchmark utility (`g2g/bench/splitpoint_tune.py`) which:
1. Runs the target system with all-CPU and all-GPU to collect per-group timings
2. Fits T_cpu and T_gpu models
3. Computes crossover P×M²
4. Sweeps SPLITPOINTS values to find the wall-time minimum
5. Outputs the recommended LIO_SPLIT_POINTS value

This should be run once per system (hardware + molecule) or when the balance
diagnostics (`verbose > 3`) show significant idle time.

### 3. Disable CPU rebalancing entirely (LOW effort, SMALL impact)

CPU rebalancing is unnecessary when CPU threads aren't the bottleneck. It adds
complexity and causes nondeterminism. For single-GPU setups, consider simply
disabling `rebalance()` entirely — the initial bin-packing in
`compute_work_partition()` is good enough, and the GPU thread determines wall
time regardless.

### 4. Pipeline CPU work during GPU idle time (HIGH effort, HIGH impact)

The real opportunity is not rebalancing but pipelining. During each GPU group's
`add_rmm_output()` (CPU scatter, ~60% of GPU thread time), the GPU is idle.
If `get_rmm_input()` for the NEXT group could run concurrently, or if
`add_rmm_output()` were moved to a helper CPU thread, the GPU thread's serial
overhead would shrink. This is tracked in `../gpu/optimize_rmm_gather_gpu.md`.

---

## Files

| File | Relevant code |
|---|---|
| `g2g/partition.cpp:431` | `rebalance()` implementation |
| `g2g/partition.cpp:502` | `solve()` — parallel region + accumulation |
| `g2g/regenerate_partition.cpp:326` | SPLITPOINTS, group creation |
| `g2g/partition.h:150` | `is_big_group()` → SPLITPOINTS threshold |
| `../gpu/optimize_memory_pool.md` | Auto-fgm caching (solves memory economics) |
| `../gpu/optimize_rmm_gather_gpu.md` | GPU-side gather/scatter (pipelining) |
| `g2g/bench/splitpoint_tune.py` | Benchmark utility for SPLITPOINTS auto-tuning |
