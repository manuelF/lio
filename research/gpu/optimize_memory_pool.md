# Automatic GPU Memory Caching — Replacing Manual `fgm` Tuning

**Status:** PHASE 1 IMPLEMENTED (opt-in via `free_global_memory = -1`)
**Priority:** HIGH — 34% wall time speedup measured on fosfatoQMMM
**Prerequisite:** fgm weights bug fixed (commit `7c58baa8`)

---

## Problem Statement

LIO's `free_global_memory` (fgm) parameter controls GPU memory caching of
basis function values across SCF iterations. With the default `fgm=0.0`,
**no caching occurs** — every iteration recomputes and reallocates all
function/gradient/hessian buffers for every GPU point group. This causes:

- **19,061 cudaMalloc + 18,780 cudaFree** calls per SCF run (fosfatoQMMM)
- **2.07 seconds** (35% of wall time) in memory management overhead
- **1,890 gpu_compute_functions** kernel launches that could be skipped
- **3,920 transpose** kernel launches that could be skipped

Setting `fgm=0.8` gives a **34% speedup** (5.87s → 3.84s on fosfatoQMMM) by
caching function values and skipping recomputation after the first iteration.

### Why fgm requires tuning (and shouldn't)

1. **System-size dependent**: fgm=0.8 works for small systems (fosfatoQMMM,
   34 QM atoms) where all ~78 GPU groups fit in 80% of GPU memory. For larger
   systems (100+ atoms, thousands of groups), 80% may not be enough.

2. **Imprecise accounting**: `GlobalMemoryPool::tryAlloc()` uses `size_in_gpu()`
   which counts only function/gradient/hessian buffers (the "big 5" cached
   matrices). But each `solve_closed()` also allocates ~15 temporary matrices
   (partial_densities, dxyz, dd1, dd2, factors, rmm_output, energy, forces,
   etc.) whose memory isn't tracked by the pool. These temporaries are cached
   by `CudaMatrix::resize()` (no realloc when dimensions match), but their
   initial allocation comes from free GPU memory that the pool doesn't know about.

3. **SPLITPOINTS interaction**: The env variable `LIO_SPLIT_POINTS` (default 200)
   controls which groups go to CPU vs GPU. More CPU groups = fewer GPU groups =
   less memory needed for caching. A user tuning fgm must also consider their
   SPLITPOINTS setting.

4. **No graceful degradation**: With fgm=0.0 (default), ALL groups are uncached.
   With fgm=0.8, groups are cached until the pool is exhausted, then remaining
   groups are uncached. But there's no prioritization — groups are cached in
   whatever order they're first encountered (which is round-robin by GPU thread,
   not by cost or reuse frequency).

---

## Current Architecture

### GlobalMemoryPool — an accounting ledger, not a pool

```
GlobalMemoryPool::init(fgm=0.8)
  → queries cudaMemGetInfo() → free_memory = 7.2 GB (on GTX 1080)
  → _freeGlobalMemory = free_memory * 0.8 = 5.76 GB

compute_functions():
  if inGlobal: return early (cached)
  if tryAlloc(size_in_gpu()) succeeds:
    inGlobal = true                    ← cache this group's function values
  allocate function_values, gradient_values, hessian_values...
  launch gpu_compute_functions + transpose kernels

solve_closed() end:
  if !inGlobal:
    function_values.deallocate()       ← throw away computed values
    gradient_values.deallocate()
    ...
```

The "pool" is purely bookkeeping — `tryAlloc` decrements a counter, `dealloc`
increments it. No actual memory is managed. The real cudaMalloc/cudaFree
happens in `CudaMatrix::resize()` and `CudaMatrix::deallocate()`.

### What gets cached when inGlobal=true

1. `function_values` — M×P scalar matrix (basis functions at grid points)
2. `gradient_values` — M×P vec4 matrix (function gradients)
3. `hessian_values_transposed` — 2M×P vec4 matrix (function Hessians)
4. `function_values_transposed` — P×M scalar (transposed layout for density)
5. `gradient_values_transposed` — P×M vec4 (transposed for density_derivs)

These are the expensive ones: they require `gpu_compute_functions` + two
`transpose` kernel launches to compute. Total = `COALESCED_DIM(P) × M × 18 × sizeof(scalar_type)` bytes per group (GGA case).

### What's NOT cached but could be (member CudaMatrix temporaries)

These are already member variables on `PointGroupGPU` and `CudaMatrix::resize()`
skips realloc when dimensions match. So they ARE effectively cached after the
first iteration — but their memory isn't accounted for by the pool:

- `partial_densities_gpu`, `dxyz_gpu`, `dd1_gpu`, `dd2_gpu`
- `factors_gpu`, `rmm_output_gpu`, `point_weights_gpu`
- `rmm_input_gpu`, `rmm_bigs_gpu`, `rmm_rows_gpu`, `rmm_cols_gpu`
- `dd_gpu`, `forces_gpu` (density_derivs/forces temporaries)
- `rmm_cuArray` (texture array — allocated once, reused)

### True per-call temporaries (stack-local CudaMatrix)

These are the actual churn — constructed and destructed every call:

| Location | Variable | Calls | Purpose |
|---|---|---|---|
| `compute_functions()` L972 | `points_position_gpu` | 1890 | Grid point positions |
| `compute_functions()` L973 | `factor_ac_gpu` | 1890 | Contraction coefficients |
| `compute_functions()` L974 | `nuc_gpu` | 1890 | Nuclear indices |
| `compute_functions()` L975 | `contractions_gpu` | 1890 | Contraction counts |
| `solve_closed()` L493 | `nuc_gpu` | 1820 | Nuclear indices (density_derivs) |
| `compute_weights()` L1118 | `atom_positions_gpu` | 70 | Atom positions |
| `compute_weights()` L1120 | `nearest_dist_gpu` | 70 | Nearest neighbor dists |
| `compute_weights()` L1122 | `atom_atom_dist_gpu` | 70 | All pairwise distances |

With `inGlobal=true` (fgm > 0), `compute_functions()` returns early →
7560 constructor/destructor pairs eliminated. But `nuc_gpu` in density_derivs
(1820 calls) and the 210 compute_weights temporaries remain.

---

## Proposed Solution: Three-Phase Approach

### Phase 1: Auto-sizing with smart default (MINIMAL CHANGE, HIGH IMPACT)

**Change default `fgm` from 0.0 to auto-detected value.**

Instead of a fixed fraction, compute the optimal cache budget at partition time:

```cpp
// In regenerate_partition.cpp, AFTER all groups are created:
size_t total_cache_need = 0;
for (auto* group : gpu_groups)
    total_cache_need += group->size_in_gpu();

size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);

// Reserve headroom for non-cached allocations (temporaries, textures, etc.)
// Empirically: ~30 MB per group for temporaries + ~200 MB system overhead
size_t headroom = gpu_groups.size() * 30 * 1024 * 1024 + 200 * 1024 * 1024;
// But don't exceed 40% of free memory for headroom
headroom = min(headroom, (size_t)(free_mem * 0.4));

size_t cache_budget = free_mem - headroom;

if (total_cache_need <= cache_budget) {
    // Everything fits — cache all groups
    effective_fgm = (double)total_cache_need / (double)free_mem;
} else {
    // Partial caching — use available budget
    effective_fgm = (double)cache_budget / (double)free_mem;
}

GlobalMemoryPool::init(effective_fgm);
```

**Key insight**: We know the exact total cache need at partition time (all groups
are already created with their `size_in_gpu()` computed). We can make an optimal
decision rather than relying on a user-supplied fraction.

**Graceful degradation**: When memory is insufficient, the pool budget limits how
many groups get cached. Groups that don't get cached still work — they just
recompute functions each iteration. The user's `fgm` parameter, if explicitly
set to a non-zero value, overrides the auto-detection.

### Phase 2: Priority-based caching (MEDIUM EFFORT, MODERATE IMPACT)

When not all groups fit in GPU memory cache, prioritize which groups to cache:

**Priority = cost_saved_per_byte = (iterations - 1) × compute_time / size_in_gpu()**

Groups with high function count (expensive `compute_functions`) and small cache
footprint should be cached first. Groups with few points/functions are cheap to
recompute and should be evicted first.

Implementation:
1. After partition, sort GPU groups by `cost() / size_in_gpu()` (descending)
2. In `compute_functions()`, process groups in priority order
3. When pool runs out, remaining groups run uncached

This is simple because groups are created once (at partition time) and their
sizes are fixed. No runtime eviction logic needed.

**Alternative (simpler)**: Sort by `size_in_gpu()` ascending (cache small groups
first). Small groups have the best cost/byte ratio because `compute_functions`
has fixed overhead per launch. This also maximizes the NUMBER of cached groups,
which matters because each uncached group has ~4 cudaMalloc+cudaFree cycles.

### Phase 3: Eliminate remaining per-call temporaries (LOW EFFORT, MINOR IMPACT)

Move stack-local `CudaMatrix` temporaries to `PointGroupGPU` member variables:

1. **`nuc_gpu` in density_derivs** (L493, L857): Change from stack-local
   `CudaMatrixUInt nuc_gpu(this->func2local_nuc)` to a member variable that's
   uploaded once in `compute_indexes()` or `compute_functions()`.
   Saves: 1820 cudaMalloc+cudaFree (closed-shell) or 1820 (open-shell).

2. **`points_position_gpu`, `factor_ac_gpu`, `nuc_gpu`, `contractions_gpu`
   in compute_functions**: These are only needed for the first call when
   `inGlobal=false`. With Phase 1 caching, `compute_functions` runs only
   once per group (first iteration). Still worth caching if fgm auto-detection
   can't cache everything. Low priority.

3. **compute_weights temporaries**: Only called 70 times (once per group at
   partition creation). Not worth optimizing.

---

## Expected Impact

### Phase 1 alone (auto-sizing)

For fosfatoQMMM-sized systems (small, everything fits):
- Same as fgm=0.8: **34% wall time speedup** (5.87s → 3.84s)
- cudaMalloc calls: 19,061 → ~2,051 (−89%)
- gpu_compute_functions eliminated after first iteration
- No user configuration needed

For large systems (partial caching):
- Automatically caches as many groups as fit
- Remaining groups run uncached (no regression vs fgm=0.0)
- Better than both fgm=0.0 (no caching) and fgm=0.8 (might OOM)

### Phase 2 (priority caching)

Only matters for large systems where partial caching occurs:
- Ensures the most cost-effective groups are cached first
- Could improve cache hit rate by 10-30% vs random caching order

### Phase 3 (eliminate temporaries)

- Saves ~1820 cudaMalloc+cudaFree per SCF run (closed-shell)
- Estimated: −50-100 ms (< 2% wall time on fosfatoQMMM)
- Primarily reduces API overhead noise in profiles

---

## Implementation Notes

### Current implementation (Phase 1)

- Default `free_global_memory = 0.0` (no caching, backward-compatible)
- `free_global_memory = -1` triggers auto-detection (opt-in)
- `free_global_memory = 0.8` uses explicit 80% budget (existing behavior)
- Auto-detection code is in `regenerate_partition.cpp`, runs after all groups are created
- Headroom = max(20% of free GPU memory, 200 MB)

### Rebalancer nondeterminism with caching

**IMPORTANT**: Enabling caching introduces run-to-run nondeterminism (~0.0002 Ha on
Fe3H2O6). Root cause: `Partition::rebalance()` (partition.cpp:431) reassigns groups
between OpenMP threads based on wall-clock timing. With caching, the GPU thread finishes
faster → different timing profile → different rebalance decisions → different CPU
group-to-thread assignments → different float64 accumulation order → slightly different
total energy.

Without caching, the timing is perfectly reproducible (same malloc/compute/free pattern
every iteration) → rebalance makes identical decisions → deterministic results.

**This is NOT a caching correctness bug.** The cached function values are identical to
recomputed values. The variation is from FP accumulation order, which is inherent to
the timing-dependent rebalancer design. It affects ALL tests equally and is within the
float32 noise floor (~1e-6 in density, ~1e-4 in total energy).

**Mitigation**: Keep `fgm=0.0` as default to preserve test determinism. Users opt-in
to auto-caching (`fgm=-1`) with awareness of the rebalancer interaction.

**Future fix**: Make `rebalance()` order-deterministic (sort groups by index after
reassignment) or use compensated accumulation for per-thread result merging.

### Accounting accuracy

The current `size_in_gpu()` underestimates actual GPU memory usage per group because
it doesn't count temporaries. However, for Phase 1 this is actually fine:
- Member-variable temporaries are allocated lazily by `CudaMatrix::resize()` on first use
- They persist across iterations (resize is a no-op when dimensions match)
- The headroom calculation accounts for this

The headroom formula `gpu_groups.size() * 30MB + 200MB` is conservative. Better:
compute actual temporary memory per group from dimensions (known at partition time).

### Thread safety

GlobalMemoryPool is accessed from a single GPU thread (gpu_threads=1 is the common
case). Multi-GPU support would need per-device atomics, but that's not the current use case.

---

## Files to Modify

| File | Change |
|---|---|
| `g2g/regenerate_partition.cpp` | Auto-detect cache budget after group creation |
| `g2g/global_memory_pool.h/cpp` | Add `initAuto(gpu_groups)` or modify `init()` |
| `g2g/init.cpp` | Keep `free_global_memory=0.0` default as "auto" sentinel |
| `g2g/cuda/iteration.cu` | (Phase 3) Move nuc_gpu to member variable |
| `g2g/partition.h` | (Phase 3) Add nuc_gpu member to PointGroupGPU |

---

## Testing Plan

1. **fosfatoQMMM (small)**: Verify auto-caching gives same speedup as fgm=0.8
2. **Convergence**: Must converge in exactly 25 iterations (no FP changes)
3. **Large system**: Test with a system that exceeds GPU memory to verify graceful degradation
4. **Explicit fgm override**: Verify that user-set fgm still works
5. **Profile comparison**: nvprof API summary showing reduced cudaMalloc/cudaFree counts
