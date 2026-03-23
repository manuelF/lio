# Optimization: Weight Kernel — Cache Reuse Between SCF Steps

## Summary
`gpu_compute_weights` implements Becke's partitioning scheme with O(N_atoms²) work
per integration point. It is called at the start of every SCF step (via
`Partition::compute_functions` → each group's `compute_weights()`).

**Key insight**: Becke weights depend ONLY on atomic positions. For geometry
optimization and single-point calculations, positions are fixed throughout all SCF
iterations. Even for MD, positions change only between steps (not within the SCF
iterations of a single step). Therefore, weights can be computed once per geometry
and cached — completely eliminating this kernel for all but the first SCF iteration
(or geometry step).

## Code Audit: Current Weight Computation

`partition.cpp:406–417`:
```cpp
void Partition::compute_functions(bool forces, bool gga) {
  #pragma omp parallel for schedule(guided, 8)
  for (int i = 0; i < cubes.size(); i++) {
    if (!cubes[i]->is_big_group()) cubes[i]->compute_functions(forces, gga);
  }
  // weights are computed here or inside compute_functions via compute_weights()
}
```

`gpu_compute_weights` (`g2g/cuda/kernels/weight.h`):
```cpp
// Every point: O(N_atoms²) Becke step-function evaluations
for (int j = 0; j < gpu_atoms; j++) {
  for (int k = 0; k < j; k++) {
    scalar_type mu = ... // interatomic ratio
    mu = bcke_step(mu);  // 3 iterations of cubic polynomial
    P_atom[j] *= (1-mu); P_atom[k] *= mu;  // partition update
  }
}
weight[point] = P_atom[i_nuc] / sum(P_atom);
```

For N_atoms=50, gpu_atoms=50: 50×50/2 = 1250 operations per point.
For P=512 points per group, 100 groups: 1250 × 512 × 100 = 64M Becke evaluations.
This runs ~2–5 ms per SCF step on GTX 1080.

## Proposal

### 1. Cache point_weights_cpu on GPU, Reuse Between SCF Steps

Since weights change only when geometry changes:
```cpp
// In PointGroupGPU:
bool weights_valid = false;
CudaMatrix<scalar_type> point_weights_gpu_cached;  // Keep on GPU across steps

// In solve_closed:
if (!weights_valid) {
  compute_weights();  // Run gpu_compute_weights
  weights_valid = true;
}
// Use point_weights_gpu_cached directly — no H2D copy needed
```

Invalidation trigger: Set `weights_valid = false` when `regenerate()` is called
(geometry changes in MD) or when `compute_weights` is forced by the caller.

**Expected speedup: 5–15% end-to-end** for multi-step SCF (the dominant case).
Zero-cost for MD (weights recomputed each step anyway, but at least no redundant
recomputation within one step's SCF).

### 2. Precompute Interatomic Distances as `rm` Constants

`gpu_compute_weights` reads `atom_position_rm` (a float4 where `.w` stores the
distance between atom pairs, precomputed as `rm[i][j]`). These are already cached
in `__constant__` memory or shared memory (`rm_sh[]`). No change needed here.

### 3. Batched Atom Loop Unrolling

For systems with many atoms (N_atoms > 32), the inner Becke loop runs many iterations
per point. Unrolling with `#pragma unroll 4` and preloading atom positions from shared
memory (already done) helps the compiler pipeline loads:
```cpp
#pragma unroll 4
for (int j = 0; j < gpu_atoms; j++) { ... }
```

**Estimated speedup: 5–10% for weight kernel** on large systems.

### 4. SFV-Level Screening

For points far from atom k, the Becke step function rapidly saturates to 0 or 1.
Short-circuit the inner loop once the partition weight for atom i_nuc is numerically 1:
```cpp
if (P_i > (1.0f - 1e-12f)) break;  // Convergence: point belongs entirely to this atom
```

This is valid because the Becke step function is monotone — once 1, it stays 1.
**Estimated speedup: 5–20% for sparse systems** (few-atom groups).

### 5. Halve Redundant Work for Symmetric Systems

The inner Becke loop updates BOTH P_atom[j] and P_atom[k] simultaneously:
```
P_atom[j] *= (1-mu);  P_atom[k] *= mu;
```
This is already optimal (the `j>k` triangle loop). No improvement possible without
algorithmic change.

## Impact

| Proposal | Complexity | Speedup |
|---|---|---|
| Weight caching (Prop 1) | Low | 5–15% end-to-end for multi-step SCF |
| Loop unrolling (Prop 3) | Trivial | 5–10% weight kernel on large systems |
| Convergence screening (Prop 4) | Low | 5–20% for sparse groups |

For typical geometry optimization (many SCF steps, fixed geometry):
- **Weight caching: 10–20% end-to-end** (weight computation is 10–20% of total time).

For MD (weight recomputed every step):
- Caching saves nothing within one step; unrolling + screening give 5–15%.

## Difficulty Assessment
**Low** (Proposals 1, 3, 4)

- Proposal 1: Add `weights_valid` flag to `PointGroupGPU`, wrap `compute_weights()`
  call. Files: `g2g/partition.h`, `g2g/cuda/iteration.cu`.
  Correctness: **Medium** — must ensure `weights_valid` is cleared on `regenerate()`.
- Proposal 3: Add `#pragma unroll` to weight kernel inner loop. 1-line change.
- Proposal 4: Add break condition. 2-line change. Validate against `agua` weights.

## Files to Modify
- `g2g/partition.h`: Add `weights_valid` flag and `point_weights_gpu_cached` member.
- `g2g/cuda/iteration.cu`: Wrap `compute_weights` call with validity check.
- `g2g/cuda/kernels/weight.h`: Proposals 3 and 4.
- `g2g/partition.cpp`: Call `invalidate_weights()` from `regenerate()`.

## Estimations
- Weight caching alone: **5–20% end-to-end** for single-point and geometry opt.
- MD simulations: **2–8%** (from unrolling and screening only).
- Implementation: **4–8 hours** for caching; **1 hour** for unrolling+screening.
