# Roofline Analysis: `gpu_compute_density<float, false>` (GGA, closed-shell)

Date: 2026-03-05
Hardware: GTX 1080 (SM 6.1, Pascal), CUDA 12.0
Test case: fosfatoQMMM (34 QM atoms, 25 SCF iterations, fgm=0.0)
Kernel: `gpu_compute_density<float, bool=0>` — the XC density accumulation kernel

## Hardware Ceilings (GTX 1080)

| Parameter | Value |
|---|---|
| FP32 peak | 8,873 GFLOP/s |
| DRAM bandwidth | 320 GB/s |
| Ridge point (FP32) | 27.7 FLOP/byte |
| L2 cache size | 2 MB |
| Registers per SM | 65,536 |
| Shared memory per SM | 48 KB |

## Measured Metrics (nvprof, 1890 invocations)

| Metric | Per invocation (avg) | Total (all 1890) |
|---|---|---|
| SP FLOPs (`flop_count_sp`) | 502M | 949 GFLOP |
| SP FMA ops | 246M | 464 GFLOP |
| SP MUL ops | 4.2M | 7.9 GFLOP |
| SP ADD ops | 7.0M | 13.2 GFLOP |
| DRAM read bytes | 30.0 MB | 56.7 GB |
| DRAM write bytes | 0.44 MB | 0.83 GB |
| DRAM total | 30.4 MB | 57.5 GB |
| L2 read transactions (x32B) | 114 MB | 215 GB |
| L2 write transactions (x32B) | 0.47 MB | 0.89 GB |
| Kernel time (avg) | 470 us | 921.6 ms |
| Achieved occupancy | 50.8% (avg) | — |
| Warp execution efficiency | 73.5% (avg) | — |
| Texture cache hit rate | 82.9% (avg) | — |
| stall_exec_dependency | 20.1% (avg) | — |
| stall_sync | 13.6% (avg) | — |

Note: FMA counts as 2 FLOPs in `flop_count_sp`. The FMA-dominated mix
(246M FMA × 2 = 492M out of 502M total) confirms this is a multiply-accumulate
workload (dot products: `w += rdm * fj_val`).

## Arithmetic Intensity

| Level | FLOPs | Bytes | AI (FLOP/byte) |
|---|---|---|---|
| DRAM | 949 GFLOP | 57.5 GB | **16.5** |
| L2 | 949 GFLOP | 216 GB | **4.4** |

Both values are below the ridge point (27.7), confirming **memory-bound** at
both DRAM and L2 levels.

## Roofline Position

```
GFLOP/s (log)
  |
  8873 |                              __________ Compute ceiling (FP32 peak)
       |                         ____/
  5280 |                    ____/ .............  Memory ceiling at AI=16.5
       |               ____/
  1030 |..........X____/  <-- Achieved (19.5% of ceiling)
       |      ____/
       | ____/
       |/____________________________________________ AI (FLOP/byte, log)
       1    4.4   10    16.5   27.7   100
            ^L2    ^DRAM-AI    ^ridge
```

## Performance Gaps

| Metric | Achieved | Ceiling | Utilization |
|---|---|---|---|
| GFLOP/s | 1,030 | 5,280 (mem-limited) | **19.5%** |
| DRAM bandwidth | 62.4 GB/s | 320 GB/s | **19.5%** |
| Occupancy | 50.8% | 56% (reg-limited, 56 regs) | 90.7% of max |

The kernel achieves only **19.5% of the memory-bound roofline ceiling**.
Occupancy is near its theoretical max (90.7%), so the gap is NOT from
insufficient warp-level parallelism alone.

## Root Cause Analysis: Why 19.5% of Ceiling?

### 1. Warp divergence (triangular loop) — ~26% wasted lanes

The inner bj-loop has condition `if (full_block || (bj+j) <= i)` which causes
threads with small `i` to be idle while threads with large `i` are active.
`warp_execution_efficiency = 73.5%` means 26.5% of warp slots are wasted.

**Impact on bandwidth:** Inactive threads still occupy warp slots but produce no
useful memory requests. Effective bandwidth = 62.4 × 0.735 = only 45.9 GB/s of
"useful" bandwidth.

### 2. Texture fetch latency (stall_exec_dependency = 20%)

The RMM density matrix is accessed via 2D texture:
```
fetch(rmm_input_gpu_tex, (float)(bj+j), (float)i)
```
Each texture fetch has ~100+ cycle latency. With 50.8% occupancy (about 16 warps
per SM on average), there are enough warps to partially hide this, but 20% of
issue slots still stall on execution dependency (waiting for texture result).

### 3. L2 traffic amplification (3.75x)

L2 sees 216 GB of read traffic but only 57.5 GB goes to DRAM (73% L2 hit rate).
The texture cache (L1TEX) hits 83% of requests, but misses still generate
significant L2 traffic. The 4.4 FLOP/byte at L2 level means L2 bandwidth is
the tighter constraint.

### 4. vec4 waste in gradient/hessian reads

`gradient_values_transposed` and `hessian_values_transposed` are stored as
`vec_type<float,4>` but only 3 components (x,y,z) are used. The 4th component
adds ~25% wasted memory traffic for these arrays.

## Register Reduction Experiments (2026-03-05)

Three approaches were tested to reduce register pressure. **All failed** due to
SCF convergence sensitivity to floating-point summation order in float32.

| Approach | Regs | Spill | Occupancy | SCF iters | Wall time | vs Baseline |
|---|---|---|---|---|---|---|
| **Baseline (dual-row, no bounds)** | 56 | 0 | 56% | 25-27 | ~12.0s | — |
| Single-row kernel (1 row/thread) | 32 | 0 | 100% | 36 | 15.6s | **-30%** |
| `__launch_bounds__(64, 24)` | 40 | 36B | 78% | 27 | 12.45s | **-4%** |
| `__launch_bounds__(64, 20)` | 48 | 0 | 66% | 32 | 14.1s | **-18%** |

### Root cause of failure

SCF convergence in float32 is extremely sensitive to FP accumulation order.
Any change to register allocation or instruction scheduling — even without
spilling — shifts the rounding pattern enough to add 5-10 SCF iterations.
The extra iterations cost more wall time than the per-kernel speedup from
higher occupancy.

**Single-row detail:** Removing the second row per thread (i2/w2/w32/ww12/ww22
accumulators) halved registers (56→32) and doubled occupancy (56%→100%), but
also doubled `block_height` (3→6 block rows). `gpu_accumulate_point` then sums
6 partial results instead of 3, changing FP order → 36 iterations (+11).

**`__launch_bounds__` detail:** Even without spilling, the compiler reorders
instructions to fit the register budget, changing which FMAs are issued in
which order → different rounding → different SCF trajectory.

### Conclusion

**Register pressure reduction is a dead end for this kernel.** The float32 SCF
solver's FP-order sensitivity makes any occupancy-improving transformation a
net performance loss. This removes opportunity A from the list below.

## Optimization Opportunities (ranked by expected impact)

### ~~A. Reduce register pressure~~ — RULED OUT (see above)

### B. Reduce warp divergence in triangular loop

The triangular access pattern `(bj+j) <= i` wastes ~26% of warp capacity.
Options:
- **Rectangular tiling:** Restructure to process rectangular sub-blocks of the
  symmetric matrix, with separate kernels for diagonal and off-diagonal tiles.
  Off-diagonal tiles have no divergence; diagonal tiles are small.
- **Warp-shuffle redistribution:** Have all threads compute, then shuffle
  results to the correct thread.
**Expected impact:** Up to 1.36x (recover the 26.5% wasted efficiency).
**Risk:** Significant code restructuring, complex to maintain.

### C. Replace texture with `__ldg()` + L2

The 2D texture path has:
- Fixed-size cache (48 KB L1TEX per SM on Pascal)
- 2D spatial locality heuristic (good for images, less ideal for matrix triangles)
- `cudaMallocArray` + `cudaMemcpy2DToArray` overhead per group

Alternative: Store RMM in regular global memory, use `__ldg()` for read-only
cached access through L2 (2 MB shared across SMs). L2 uses LRU eviction which
may better match the sequential bj-block access pattern.
**Expected impact:** Unclear; texture hit rate is already 83%. Could help or hurt.
**Risk:** Need to profile both paths.

### D. Eliminate vec4 waste in gradient/hessian arrays

Store gradients as SoA (separate x, y, z arrays) instead of AoS (vec4).
This eliminates the 25% wasted bandwidth from unused w component.
**Expected impact:** ~5-10% DRAM bandwidth savings for GGA path.
**Risk:** Requires changing the transpose kernel and function evaluation layout.

### E. Fuse density + accumulate_point kernels

Currently two separate kernel launches: `gpu_compute_density` produces partial
block-row results, then `gpu_accumulate_point` sums them and applies XC.
Fusing would eliminate the intermediate `partial_densities_gpu` write+read.
**Expected impact:** Reduces DRAM traffic but adds complexity.
**Risk:** `gpu_accumulate_point` has different block geometry (256 threads, 1D).

## Key Takeaway

This kernel is **latency-limited within a memory-bound regime**. It uses only
19.5% of the memory-bound ceiling, not because of insufficient parallelism
(occupancy is 91% of max), but because of:
1. Triangular loop divergence wasting 26% of warp capacity
2. Texture fetch latency not fully hidden at 50.8% occupancy
3. L2 being the real bandwidth bottleneck (AI = 4.4 at L2 level)

**Critical constraint:** Float32 SCF convergence is extremely sensitive to FP
summation order. Register pressure reduction (opportunity A) has been ruled out
experimentally — all three tested approaches (single-row, `__launch_bounds__`
with and without spilling) caused 5-11 extra SCF iterations, making them net
performance losses. Any remaining optimization must preserve the exact FP
accumulation order or accept the risk of shifted convergence.

The most impactful **safe** optimization would be replacing texture with
`__ldg()` (opportunity C), as it could improve L2 utilization without changing
any FP accumulation order. Warp divergence reduction (opportunity B) has the
highest theoretical ceiling but carries the same FP-order risk since it
restructures the accumulation loop.

## Profiling Commands Used

```bash
# Collect FLOP + memory metrics (filtered to density kernel)
nvprof --kernels "gpu_compute_density" \
  --metrics flop_count_sp,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_add,\
dram_read_bytes,dram_write_bytes,l2_read_transactions,l2_write_transactions \
  liosolo -i fos.in -c fos.xyz -b basis 2> roofline_metrics.txt

# Occupancy + stall metrics (from earlier collection)
nvprof --metrics achieved_occupancy,warp_execution_efficiency,\
stall_exec_dependency,stall_sync,tex_cache_hit_rate \
  liosolo -i fos_fgm.in -c fos.xyz -b basis 2> metrics_density.txt

# Kernel time summary
nvprof -o profile.nvvp liosolo -i fos.in -c fos.xyz -b basis
nvprof -i profile.nvvp --print-gpu-summary
```
