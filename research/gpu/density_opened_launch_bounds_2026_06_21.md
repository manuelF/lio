# Open-shell density kernel: __launch_bounds__ register cap — DONE 2026-06-21

## Summary

`gpu_compute_density_opened` (the open-shell XC density kernel, `g2g/cuda/kernels/energy_open.h`)
had **no `__launch_bounds__`** — unlike its closed-shell twin `gpu_compute_density`
(`energy.h`), which has carried `__launch_bounds__(DENSITY_BLOCK_SIZE, 16)` for a long time.
That asymmetry let ptxas spend **78 registers/thread**, capping the kernel at **12 blocks/SM**
on Ampere (granularity rounds 78→80; 65536/(80·64)=12).

Added `__launch_bounds__(DENSITY_BLOCK_SIZE, 14)` → ptxas targets **71 regs**, **14 blocks/SM**,
**zero spills** (verified in SASS for both instantiated variants `<float,false,false>` and
`<float,false,true>`; only GGA/`lda=false` variants are instantiated for open-shell).

## Why 14, not 16

`(…,16)` forces a 64-reg target and **spills** (23 STL/LDL ops in the `<false,false>` SASS) —
the open kernel does ~2× the live-state work of closed (α+β accumulators), so 64 regs is too
tight. `(…,14)` (71–72 regs) is the sweet spot: one occupancy step up, no spills. This is also
why the earlier blanket `__launch_bounds__(64)` attempt was logged as "net-neutral/regression"
([[density_opened_point_batch_dead_end_2026_05_25]]) — it was the 64-reg (spilling) variant.

## Measured (multiZn / 12_Zn_timers, RTX 3080 Ti, hybrid float build)

Profiled the **heavy** launches (grid 706–726 = block_height=2, group_m 129–256), which dominate
total kernel time:

| Metric (heavy `<false,false>` launches) | Baseline (78 reg) | `(64,14)` (71 reg) |
|---|---|---|
| Occupancy limit (registers) | 12 blocks/SM | 14 blocks/SM |
| Achieved warps active | 18% | **36%** (2×) |
| SM throughput | 22% | **38%** |
| Density kernel total GPU time (nsys, full run) | 157.6 ms | **140.6 ms (−10.8%)** |

The heavy launches have enough blocks (726 over 80 SMs ≈ 9 blocks/SM) to be genuinely
register-bound, so lifting the cap 12→14 nearly doubles their achieved occupancy. The *small*
launches (grid ≈353, group_m 65–128, block_height=1) stay grid-limited (4.4 blocks/SM) and are
unaffected — but they are the cheap ones.

The kernel is **latency-bound** (SM 22% / DRAM 5% at baseline), so the occupancy bump directly
buys latency hiding, not bandwidth.

## Wall impact & where it matters

On the **3080 Ti, multiZn total wall is unchanged** (22.4→22.5s, noise) — multiZn is CPU-BLAS
bound (diag + DIIS/BChange accel ≈ 4 s/iter vs g2g 145 ms/iter; g2g overlap timer 148→145 ms).
This matches [[overlap_bchange_serialized_2026_06_21]]: multiZn's real poles are CPU BLAS, not g2g.

The win lands where **g2g is the actual pole**: GTX-1080-class / SM-starved GPUs and large
systems (see [[partition_geometry_megakernel_dead_end_2026_05_31]] caveat, and
[[fosfato_g2g_exposed_pole_2026_06_19]]). On those, the dominant XC density kernel running at
−10.8% GPU time / 2× occupancy translates much more directly to wall. Per the project owner's
"don't overfit for the last 2-3 GPUs" rule, this is a no-toggle, hardware-portable improvement
that helps older hardware most and is neutral-to-slightly-positive on Ampere.

## Correctness

`__launch_bounds__` changes register allocation only — **no arithmetic reorder → bit-exact by
construction**. Validated: **02_Fe3H2O6 (open-shell, the float32 sensitivity guard) ALL PASS**;
00_agua PASS. 03_fosfatoQMMM Energy off 150 µHa = the documented closed-shell
`free_global_memory=-1` bistability tripwire ([[density_blas3_dead_end_2026_06_19]]) — fosfato is
**closed-shell** and never calls the changed kernel, so this is pre-existing, not from this edit.

## Build note (cost me a detour)

Do NOT set `CUDA_HOME=/usr/local/cuda-13.1` for the g2g build on this machine. Bare `nvcc` is
CUDA 12.0 and emits `cudaGetDeviceProperties_v2`, which **cudart 13 does not export** (13 only has
`cudaGetDeviceProperties`) → runtime `undefined symbol` once libg2g.so relinks against cudart 13.
Build with `make -j cuda=1 cpu=1` and CUDA_HOME unset → `-lcudart` resolves to the system
multiarch cudart 12 (`/usr/lib/x86_64-linux-gnu/libcudart.so.12`), which has the `_v2` symbol.
The stale [[project_optimizations_porting_2026_05_08]] note (CUDA_HOME=13.1) is wrong for g2g on
this branch/runtime.
