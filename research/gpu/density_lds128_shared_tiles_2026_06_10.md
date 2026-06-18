# Density kernels: 4-wide shared tiles (LDS.128) — DONE 2026-06-10

## Summary

`gpu_compute_density` (closed) and `gpu_compute_density_opened` were **LSU-bound**,
not compute- or texture-bound. The inner j-loop loaded its shared tile as
1 scalar + three 12-byte `vec_type<scalar,3>` structs ≈ 10 scalar LDS
instructions per iteration, against only ~20 FMAs. Widening the shared arrays
to `vec_type<scalar,4>` (16 B, inherits `float4`) and reading them as full
struct copies makes the compiler emit `LDS.128`, cutting shared-load
instructions per j-iteration from ~10 to 4.

**Bit-exact** (same values, same arithmetic order — only load width changes).

## Measurements (fosfato, RTX 3080 Ti, `LIO_OVERLAP_INT3LU_G2G=1`)

| Metric | Before | After |
|---|---|---|
| `gpu_compute_density` total GPU time | 283 ms (68.6% of GPU) | 199 ms (−29.5%) |
| g2g per-iter critical path | 14.3 ms (pole, idle 2.3 ms) | 11.2 ms (hidden under int3lu 12.3 ms, idle 1.1 ms) |
| Wall (median of 5) | 1.226 s | ~1.14 s (−7%) |
| `sm__inst_executed_pipe_lsu` (grid 3622×2) | 88.9% (saturated) | 48.6% |
| stall mio_throttle | 23.5% | 1.5% |
| `sm__inst_executed_pipe_fma` | 20.5% | 27.4% |
| Registers closed `<float,false>` | 56 | 61 (≤64 launch_bounds cap, occupancy unchanged) |
| Registers opened `<float,false,false>` | 80 | 72 (occupancy improves 12→14 blocks/SM) |

Energy: −2148.6509794 vs baseline −2148.6509796 A.U. — 2e-7 Ha shift is the
known fgm-caching/rebalance timing nondeterminism (fos.in has fgm=0.3), not a
kernel numerical change. Kernel unit tests (`test/unit_tests/kernels/
energy_test`, `energy_open_test`) PASS **bit-exact vs CPU reference**;
racecheck clean on both.

## How it was found

ncu on one SCF iteration's 70 launches: `sm__throughput` 90–96% on the
dominant launches but FMA only ~23% → pipe breakdown showed LSU at 89%.
The shared tile is read by all 64 threads at the same j (broadcast, no bank
conflicts) — the cost was purely instruction count, not conflicts.

## Implementation notes

- `vec_type<scalar,4>` inherits `float4`/`double4` → 16/32 B alignment in
  shared; `float4`/`double4 operator+=` in `cuda_extra.h` keeps the reduction
  tree compiling unchanged.
- Inner loop must copy the **whole** struct to a local
  (`const vec_type<scalar_type,4> fgj4 = fgj_sh[j];`) — constructing a vec3
  straight from the shared ref would emit 3×LDS.32.
- `vec_type<double,4>(double3)` ctor is `explicit` and does not set `w` →
  reduction stores use the explicit 4-arg ctor with `w=0`.
- Tile fill also got cheaper: `gradient/hessian_values` are already
  `vec_type<scalar,4>` in global, so the fill is now LDG.128→STS.128 with no
  narrowing.
- Shared usage 2560 → 3328 B/block: not an occupancy limiter (reg-limited).
- Verified in SASS: inner loop now 48×LDS.128 + 16×LDS.32 (was all scalar).

## Follow-up (same day): fj-in-w packing + warp-shuffle epilogue

Second pass, also bit-exact, two changes:

1. **fj packed into the `.w` lane of the gradient tile** — the GGA inner loop
   now does 3×LDS.128 per j instead of 3×LDS.128 + LDS.32 (−25% hot-loop
   shared-load instructions). nvcc then eliminates `fj_sh` entirely for the
   GGA instantiation (shared 3328 → 3072 B).
2. **Reduction epilogue rewritten on warp shuffles** — same addition pairs in
   the same order (cross-warp step (p, p+32) through one shared exchange,
   warp-0/left operand order preserved; the five intra-warp steps via
   `__shfl_down_sync`). Closed: 8 `__syncthreads` → 2. Opened: the α and β
   trees (~16 barriers, sequential) now reduce **concurrently** — warp 0
   folds α while warp 1 folds β — with 2 barriers total; lane 0 writes α,
   lane 32 writes β.

Results: density kernel 199 → 175 ms (−12.4%; cumulative −38% vs the 283 ms
start), g2g/iter 11.2 → 10.4 ms. Wall flat (~1.15 s) because int3lu (12.8 ms)
is the overlap pole — the win pays in non-overlapped XC contexts,
open-shell systems, and compounds with future int3lu reductions.
SASS: 4 BAR.SYNC (was ~10), 50 SHFL, zero scalar LDS in the GGA loop.
Registers: closed 61 (unchanged), opened 72 → 78 (still < original 80).
Unit tests bit-exact vs CPU, racecheck clean.

## Pass 3 (same day): both remaining hypotheses tested and REJECTED

Post-pass-2 profile: LSU down to 40–46%, issue-active 59–74%, occupancy 62%
(structural max is 67%: 64-thread blocks × SM86's 16-block/SM cap = 32 of 48
warps). Remaining stalls: long_scoreboard 20–30% (tex latency),
not_selected ~27%, barrier 3–26% (launch-dependent).

1. **Tile double-buffering — REJECTED, measured regression (+17%).**
   2-buffer shared tiles, 1 `__syncthreads`/tile instead of 2. Density kernel
   175 → 205 ms. Cause: shared 3072→6144 B/block drops the shared-mem
   occupancy limit to 14 blocks (warps_active 62% → ~50%), and the in-order
   STS-after-LDG fill stalls show up as long_scoreboard 29–40%. The barrier
   saved is worth less than the occupancy lost. **The kernel is
   occupancy-sensitive; do not trade shared footprint for barriers.**
   (Would be even worse on Turing: 64 KB shared/SM → 10 blocks.)

2. **`#pragma unroll 8` on the j-loop — REJECTED, flat (174.5 vs 175 ms).**
   ptxas already unrolls 4×; at the 64-reg `__launch_bounds__` cap deeper
   unroll just spills (STACK:8) instead of adding tex fetches in flight.

3. **closed-kernel `single_pointer` specialization — NOT WORTH IT.** Measured
   group_m distribution (gather-grid correlation): m ≤ 64 groups are only
   ~6.7% of density kernel time on fosfato. The heavy launches are
   m ∈ [64,128) spheres (gy=1, ~35%) and m ∈ [128,256) cubes (gy=2, ~55%).

**Conclusion: at 175 ms the kernel is at its practical local optimum for the
fixed geometry** (DENSITY_BLOCK_SIZE=64 is load-bearing for FP reproduction;
block-size changes alter the reduction order → forbidden). The remaining
structural steps are (a) density-as-GEMM — changes FP summation order,
forbidden by convergence constraints (see tensor-core/partition-geometry
rejections), and (b) multi-group fused launches — capped at the ~10% burst
idle (measured: 8.5 ms busy / 9.4–9.8 ms span per iteration), previously
rejected for heme on reward/complexity.

## Where further density gains would pay (unchanged)

On fosfato g2g (10.4 ms/iter) hides under int3lu (12.8 ms); gains pay in
non-overlapped XC contexts (Finalize XC energy ~12 ms, XC gradients ~20 ms)
and on open-shell/heme where density_opened is 85% of GPU time.
