# `gpu_compute_density_opened` two-regime profile — template-constants hypothesis tested

**Date:** 2026-06-22
**Status:** REJECTED (for multiZn / newest GPUs) — documents constraint + where it *would* pay
**Case:** `test/LIO_test/12_Zn_timers` (multiZn, open-shell GGA, 100 atoms, M=2600, nmax=2,
`free_global_memory=-1`, `LIO_OVERLAP_INT3LU_G2G=1`). HW: RTX 3080 Ti (80 SM, sm_86), CUDA 12.0.

## Question

Can the biggest CUDA kernel reduce register pressure by **moving runtime constants into the
template** (à la `lda`/`single_pointer`) to raise occupancy and speed XC?

## Biggest kernel (nsys)

`gpu_compute_density_opened<float, lda=false, single_pointer=false>` = **53% of all GPU time**
(140.8 ms over 1152 launches). 71 registers, capped by the existing `__launch_bounds__(64,14)`
(0 spills). `single_pointer=true` twin is 55 reg but only fires for group_m ≤ 64.

## The kernel runs in TWO regimes (the averages hide this)

| Regime | Grid | Waves/SM | Theoretical occ | Achieved occ | Share of kernel time |
|---|---|---|---|---|---|
| **Large groups** | ~5432 blk | **4.85** | 58.3% | **55.7%** (≈ at ceiling) | **83% (117 ms)** |
| Small groups | ~327 blk | 0.29 | 58.3% | 17% (grid-starved) | 17% (24 ms) |

(split measured from the nsys SQLite duration histogram: 576 launches ≥100 µs = 117 ms;
576 launches <100 µs = 24 ms. median 126 µs, p90 239 µs, max 248 µs.)

- **Large launches (83% of the time)** are pinned at the **register-limited** 58.3% occupancy
  ceiling — `Block Limit Registers = 14` < `Block Limit SM = 16`. They are **latency-bound**:
  ~50% of warp cycles stall on a scoreboard dependency waiting for the `tex2D` RDM fetches
  (Compute SM ~20%, DRAM ~5%, L1 hit 95%, L2 73%). More resident warps *would* hide that
  latency → register reduction is directionally correct **here**.
- **Small launches** are grid-starved: 327 blocks vs 80 SM × 14 blk = 1120 block slots for one
  wave → 0.29 waves/SM. The GPU is <30% filled regardless of per-SM occupancy. Register/
  occupancy changes do nothing for these.

## Why the proposed mechanism has no target

The kernel's only scalar runtime args are `m` (group_m) and `points`. Both vary launch-to-launch
across hundreds of groups → **cannot be compile-time template parameters**. The two flags that
*can* be templated (`lda`, `single_pointer`) already are. The 71 registers are the **dual-spin ×
dual-pointer GGA accumulator working set** (~40 live regs: `w / w3 / ww1 / ww2` × {a,b} × {i,i2})
plus loop temporaries — **algorithmic pressure, not constant-driven**. There is no "constant to
move into the template." Reaching 64 reg (→16 blk/SM, 67% occ) needs an algorithmic change:
drop the `i2` 2× unroll, or split α/β into two passes — both **reorder FP accumulation**.

## Decisive caveat — on this HW the kernel is hidden

Total GPU work ≈ **265 ms during a 21.5 s run (~1.2% of wall)**. multiZn is **CPU-BLAS bound**:
per-iteration poles are `BChange AOtoON` 2.89 s, `Fock diagonalization` 3.09 s, `DIIS commut`
0.92 s. The density kernel runs overlapped and finishes long before the CPU iteration work.
Halving it saves ≈0% wall on the 3080 Ti. (Consistent with [[density_opened_launch_bounds_2026_06_21]]
and [[partition_geometry_megakernel_dead_end_2026_05_31]].)

## Verdict

Do **not** ship a register change for multiZn: the proposed lever (template constants) has no
target; the real lever (de-unroll / spin-split) reorders FP on an open-shell SCF whose iter
count is Lyapunov-chaotic (judge on OMP-median, not one run); wall payoff on this GPU is 0.

**Where it WOULD pay** (per the "don't overfit to newest GPUs" rule): on older/SM-starved GPUs
(GTX 1080, ~20 SM) the large launches have even more waves/SM and the XC kernel becomes the
exposed pole instead of being hidden. There, lifting occupancy 58→67% to hide the texture-fetch
latency is a genuine win. Actionable experiment: collapse the `single_pointer=false` (71-reg)
accumulator footprint toward the `single_pointer=true` (55-reg) one, validated for kernel-local
speed AND bit-exactness / OMP-median iteration count.
