# Roofline Analysis: `gpu_compute_density<float, false>` (GGA, closed-shell)

**Status:** REF — analysis complete, opportunities re-evaluated against tested dead ends
**Last updated:** 2026-03-28
**Hardware:** GTX 1080 (SM 6.1, Pascal), CUDA 12.0
**Test case:** fosfatoQMMM (34 QM atoms, 25 SCF iterations)

## Hardware Ceilings (GTX 1080)

| Parameter | Value |
|---|---|
| FP32 peak | 8,873 GFLOP/s |
| DRAM bandwidth | 320 GB/s |
| Ridge point (FP32) | 27.7 FLOP/byte |
| L2 cache size | 2 MB |
| L1TEX cache per SM | 48 KB |
| Registers per SM | 65,536 |
| Shared memory per SM | 48 KB |

## Measured Metrics (nvprof, fgm=0.0, 1890 invocations)

**Note:** These metrics were collected with `fgm=0.0` (no function caching). With
`fgm=-1` (current default), invocation count drops to 1134 (42 groups × 27 passes),
but **per-invocation metrics are identical** — the kernel does the same work per call
regardless of caching mode. Total kernel time drops proportionally.

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
| Kernel time (avg) | 470 µs | 921.6 ms |
| Achieved occupancy | 50.8% (avg) | — |
| Warp execution efficiency | 73.5% (avg) | — |
| Texture cache hit rate | 82.9% (avg) | — |
| stall_exec_dependency | 20.1% (avg) | — |
| stall_sync | 13.6% (avg) | — |

**Current state (fgm=-1):** 602ms total, 1134 calls, 531µs avg — the ~13% increase
in per-call time vs fgm=0.0 is likely due to different group sizes after rebalancing.

FMA-dominated mix (246M FMA × 2 = 492M out of 502M total) confirms this is a
multiply-accumulate workload (dot products: `w += rdm * fj_val`).

## Arithmetic Intensity

| Level | FLOPs | Bytes | AI (FLOP/byte) |
|---|---|---|---|
| DRAM | 949 GFLOP | 57.5 GB | **16.5** |
| L2 | 949 GFLOP | 216 GB | **4.4** |

Both values below the ridge point (27.7) → **memory-bound** at both levels.

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

The kernel achieves only 19.5% of the memory-bound roofline ceiling. But
occupancy is near its theoretical max, so the gap is NOT from insufficient
warp-level parallelism alone.

## Root Cause Analysis: Why 19.5% of Ceiling?

### 1. Warp divergence (triangular loop) — ~26% wasted lanes

The inner bj-loop condition `if (full_block || (bj+j) <= i)` causes threads
with small `i` to be idle while threads with large `i` are active.
`warp_execution_efficiency = 73.5%` → 26.5% of warp slots wasted.

**Impact on bandwidth:** Inactive threads still occupy warp slots but produce
no useful memory requests. Effective bandwidth = 62.4 × 0.735 = only 45.9 GB/s
of "useful" bandwidth.

### 2. Texture fetch latency (stall_exec_dependency = 20%)

The RMM density matrix is accessed via 2D texture:
```
fetch(rmm_input_gpu_tex, (float)(bj+j), (float)i)
```
Each texture fetch has ~100+ cycle latency. With 50.8% occupancy (~16 warps per
SM on average), there are enough warps to partially hide this, but 20% of issue
slots still stall waiting for texture results.

### 3. L2 traffic amplification (3.75x)

L2 sees 216 GB of read traffic but only 57.5 GB goes to DRAM (73% L2 hit rate).
The texture cache hits 83% of requests, but misses generate significant L2
traffic. AI=4.4 at L2 level means L2 bandwidth is the tighter constraint.

### 4. vec4 waste in gradient/hessian reads

`gradient_values_transposed` and `hessian_values_transposed` are stored as
`vec_type<float,4>` but only 3 components (x,y,z) are used. The 4th component
adds ~25% wasted memory traffic for these arrays.

## Realistic Performance Ceiling

The theoretical 5.28 TFLOP/s ceiling (at DRAM AI=16.5) is unreachable because:

| Factor | Reduction | Remaining ceiling |
|--------|-----------|-------------------|
| Warp divergence (26.5% waste) | ×0.735 | 3,881 GFLOP/s |
| Texture latency at 50.8% occupancy | ×0.60 (est.) | 2,329 GFLOP/s |
| L2 as real bottleneck (AI=4.4) | further constrained | ~1,500-2,000 GFLOP/s |

**Realistic ceiling: ~1,500-2,000 GFLOP/s.** Current: 1,030 GFLOP/s = **52-69%
of realistic ceiling.** The kernel is **reasonably well-optimized** for Pascal +
float32 constraints, not 5x underperforming as the raw roofline suggests.

## Kernel Structure (energy.h) — Reference

```
Grid:   dim3(npoints, block_height=ceil(M/(2×64)))
Block:  dim3(64) = 2 warps

Each thread processes 2 function indices: i and i2 = i + 64
For each bj-block of 64:
  Load function values fj_sh[0..63] into shared memory (coalesced)
  Load gradient/hessian values into shared memory (GGA only)
  For j = 0..63:
    if (bj+j) <= i:
      rdm = tex2D(rmm, bj+j, i)        ← texture fetch (~100 cycle latency)
      w  += rdm * fj_sh[j]              ← density accumulator
      w3 += fgj_sh[j] * rdm             ← gradient accumulators (GGA)
      ww1 += fh1j_sh[j] * rdm           ← Hessian accumulators (GGA)
      ww2 += fh2j_sh[j] * rdm
    same for i2 (second row per thread)

After all bj-blocks:
  Combine i and i2 partial results
  Two-warp reduction via shared memory + warpReduceScalar
  Lane 0 writes partial_density, dxyz, dd1, dd2 to global memory
```

**Register budget (56 total):**
- Row 1 accumulators: w, w3.xyz, ww1.xyz, ww2.xyz = 10 registers
- Row 2 accumulators: w2, w32.xyz, ww12.xyz, ww22.xyz = 10 registers
- Final results: partial_rho, dxyz.xyz, dd1.xyz, dd2.xyz = 10 registers
- Temporaries + addresses + loop variables: ~26 registers

**The dual-row design (i + i2) halves block_height**, reducing intermediate
storage for `gpu_accumulate_point` and reducing grid launch overhead. It also
doubles the work per thread, helping hide texture latency.

## Optimization Opportunities — Status

### A. Reduce register pressure — RULED OUT

Three approaches tested, all failed due to float32 FP-order sensitivity:

| Approach | Regs | Occupancy | SCF iters | Wall time | vs Base |
|---|---|---|---|---|---|
| **Baseline (dual-row)** | 56 | 56% | 25-27 | ~12.0s | — |
| Single-row (1 row/thread) | 32 | 100% | 36 | 15.6s | **-30%** |
| `__launch_bounds__(64, 24)` | 40 | 78% | 27 | 12.45s | **-4%** |
| `__launch_bounds__(64, 20)` | 48 | 66% | 32 | 14.1s | **-18%** |

**Root cause:** Any change to register allocation or instruction scheduling
shifts the float32 rounding pattern. Even without spilling, the compiler
reorders FMAs to fit the register budget → different accumulation order → 5-11
extra SCF iterations → net wall time loss.

**This is a dead end.** Do not re-attempt without moving to full double precision.

### B. Reduce warp divergence in triangular loop — RULED OUT

The triangular access pattern `(bj+j) <= i` wastes 26.5% of warp capacity.

**Why this can't be fixed under float32 constraint:**

Any restructuring of the triangular loop changes which thread accumulates which
(i,j) pair and in what order. Tested approaches:

- **Rectangular tiling** (diagonal + off-diagonal blocks): changes j-loop
  traversal order → different FP accumulation → same SCF convergence risk as
  register reduction
- **Warp-shuffle redistribution**: changes which lane holds which partial sum →
  different reduction tree → different FP result
- **Load-balanced thread mapping** (triangular-to-rectangular index remap):
  changes per-thread accumulation sequence → different FP result

All these produce **numerically different float32 values** that would shift the
DIIS trajectory, likely adding SCF iterations (as observed in the register
experiments). The divergence loss is structural under the current precision model.

### C. Replace texture with `__ldg()` — REJECTED (2026-03-20)

**Tested and measured:** 36% regression in `gpu_compute_density` on Pascal.
See `optimize_density_texture.md` for full metrics.

- tex2D L1 cache hit rate: 82.85%
- `__ldg` L1 cache hit rate: 76.48%
- The 6.4pp drop causes 9.5pp more memory stall cycles → 36% wall time increase
- tex2D's 2D Morton-order tiling is a genuine hardware advantage for this access pattern

**DO NOT re-attempt on Pascal SM 6.1.** May be worth retesting on Volta+ (SM 7.0+)
where L1 is 128KB with different caching policies.

### D. Eliminate vec4 waste in gradient/hessian arrays — NOT WORTH IT

Store gradients as SoA (separate x, y, z arrays) instead of AoS (vec4) to
eliminate 25% wasted bandwidth from the unused w component.

**Why marginal:** The kernel is at 19.5% of bandwidth ceiling — it's
**latency-limited, not bandwidth-limited**. Reducing bandwidth demand by ~15%
(vec4 waste affects gradient/hessian arrays, not function values or RMM) doesn't
improve performance when the bottleneck is texture fetch latency and warp
divergence, not raw bandwidth saturation.

**Estimated impact:** <2% wall time improvement for fosfatoQMMM.
**Touches:** transpose kernel, compute_functions, all density/RMM kernels.
**Verdict:** High code churn for negligible gain at current utilization levels.
Would matter more on bandwidth-saturated hardware or larger basis sets where
the working set exceeds L2.

### E. Fuse density + accumulate_point kernels — NOT WORTH IT

`gpu_accumulate_point` takes only **3.5µs per call** (0.3% of GPU time). The
intermediate data between density and accumulate_point:
- `partial_densities`: block_height × npoints × sizeof(float)
- `dxyz`, `dd1`, `dd2`: block_height × npoints × sizeof(vec4)
- For M=86, npoints=500: block_height=1, total ~10KB per group (trivial)

Fusing would add the XC functional evaluation (calc_ggaCS: exp, log, pow) to an
already register-heavy kernel (56 regs), pushing to ~80+ registers → occupancy
drops from 56% to ~25% → severe latency hiding degradation.

**Verdict:** Saves ~3ms total (3.5µs × 1050 calls minus launch overhead savings),
costs significant occupancy. Net negative.

### F. GEMM reformulation — HIGHEST POTENTIAL, CARRIES FP-ORDER RISK

**See `optimize_density_gemm.md` for full proposal.**

Replace the custom tiled kernel with cuBLAS SGEMM:
```
Y_k(p) = Σ_j R_kj × F_j(p)    →  SGEMM: Y = R · F  (M×M · M×P → M×P)
ρ(p) = Σ_k F_k(p) × Y_k(p)    →  element-wise dot product
```

For GGA (gradients + Hessians): need 1 + 3 + 6 = 10 GEMMs total:
- 1 SGEMM for Y = R · F (density)
- 3 SGEMMs for Y'_{x,y,z} = R · ∂F/∂{x,y,z} (gradients)
- 6 SGEMMs for Y''_{...} = R · ∂²F/∂...∂... (Hessians)

**Advantages:**
- cuBLAS SGEMM achieves 70-80% of peak FLOPS via optimized tiling/double-buffering
- Arithmetic intensity O(M) per GEMM vs O(1) per texture fetch in current kernel
- Eliminates warp divergence entirely (rectangular GEMM, no triangular condition)
- Eliminates texture infrastructure (cudaArray, Memcpy2DToArray, texture objects)

**Concerns:**
1. **FP order:** cuBLAS uses internal tiling that differs from the current
   accumulation order. Float32 results WILL be numerically different. The DIIS
   impact is unpredictable without testing.
2. **Small-M overhead:** For M=30 (typical cube groups), each SGEMM is a 30×30 ×
   30×500 = 450K FLOP problem. cuBLAS launch overhead (~5-10µs) may dominate at
   this size, making it slower than the custom kernel for small groups.
3. **10 SGEMM calls:** The overhead of 10 kernel launches per group (×42 groups ×
   25 iters = 10,500 launches) could add ~50-100ms of launch overhead.
4. **Memory for intermediates:** 10 M×P matrices = 10 × 86 × 500 × 4 = 1.7MB per
   group (fits in GPU memory but adds allocation pressure).

**Expected impact:**
- For large groups (M > 100): potentially 2-5x faster per kernel invocation
- For small groups (M < 50): likely slower due to launch overhead
- Net effect depends on group size distribution and DIIS convergence behavior
- **Must be tested empirically** — cannot be predicted from roofline alone

**Verdict:** The only remaining optimization with potential for >10% wall time
improvement. But it requires careful empirical validation of both performance and
SCF convergence. Consider prototyping on the largest GPU groups first.

### G. Shared memory RMM tiling — NOT VIABLE

Loading RMM blocks into shared memory instead of texture would reduce texture
latency stalls. But the shared memory cost is prohibitive:

- RMM tile: DENSITY_BLOCK_SIZE² × sizeof(float) = 64×64×4 = **16KB**
- Existing shared: fj_sh + fgj_sh + fh1j_sh + fh2j_sh = ~2.5KB
- Total: **18.5KB per block**
- SM 6.1 allows 48KB shared per SM → only **2 blocks per SM**
- Occupancy: 2 × 64 / 2048 = **6.25%** (vs current 50.8%)

The ~8x occupancy drop would cause catastrophic latency hiding degradation,
far outweighing the reduced per-access latency.

### H. Multi-point batching — SPECULATIVE

Process K points per block instead of 1. The RMM values `R(bj+j, i)` are
identical across all points — only function values differ. Batching K points
per thread would amortize the texture fetch cost across K points.

**Problem:** Each additional point requires its own set of accumulators
(w, w3, ww1, ww2 = 10 registers). With K=2: 20 extra registers → 76 total →
occupancy drops from 56% to ~33%. The reduced texture fetches may not compensate
for the occupancy loss.

**Alternatively:** Process K points per *block* with shared memory for function
values. K=4 points × 2.5KB/point = 10KB shared. Occupancy: 48KB/10KB = 4
blocks per SM → 256 threads → 12.5% occupancy. Still too low.

**Verdict:** Not viable under current register/shared memory constraints.

## Key Takeaway

This kernel is **latency-limited within a memory-bound regime**. It achieves
52-69% of the **realistic** performance ceiling (accounting for structural warp
divergence, texture latency, and L2 amplification).

**The remaining gap cannot be closed without either:**
1. **Algorithmic change** (GEMM reformulation — opportunity F) — the only path
   to >10% improvement, but carries float32 FP-order risk
2. **Precision change** (full double, `FULL_DOUBLE` macro) — eliminates the
   FP-order sensitivity that blocks opportunities A, B, and F
3. **Hardware upgrade** (SM 7.0+) — larger L1 cache, ncu profiling, potentially
   competitive `__ldg` performance

For the current hardware + precision combination, **the kernel is near-optimal**.
The 19.5% of raw roofline is misleading — the realistic ceiling is much lower
due to structural factors that cannot be changed without breaking SCF convergence.

## Comparison: GPU vs CPU Density Kernel

The CPU kernel (`cpu_compute_density_gga` in `cpu/cpu_kernels.h`) uses **loop
fission** — splitting the 10-accumulator inner loop into 3 sub-loops of 4/3/3
accumulators, separated by `asm volatile("" ::: "memory")` barriers. This
manages x86-64 register pressure (16 YMM registers).

The GPU kernel does NOT need fission — it has 56 registers per thread, enough
for all 10 accumulators. The dual-row design (i + i2) is the GPU equivalent of
the CPU's fission: it increases work per thread to amortize fixed costs (shared
memory loads, texture fetches) while staying within the register budget.

**Both kernels are well-adapted to their respective architectures.**

## Profiling Commands Used

```bash
# Collect FLOP + memory metrics (filtered to density kernel)
nvprof --kernels "gpu_compute_density" \
  --metrics flop_count_sp,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_add,\
dram_read_bytes,dram_write_bytes,l2_read_transactions,l2_write_transactions \
  liosolo -i fos.in -c fos.xyz -b basis 2> roofline_metrics.txt

# Occupancy + stall metrics
nvprof --metrics achieved_occupancy,warp_execution_efficiency,\
stall_exec_dependency,stall_sync,tex_cache_hit_rate \
  liosolo -i fos_fgm.in -c fos.xyz -b basis 2> metrics_density.txt

# Kernel time summary
nvprof -o profile.nvvp liosolo -i fos.in -c fos.xyz -b basis
nvprof -i profile.nvvp --print-gpu-summary

# NOTE: ncu (Nsight Compute) requires SM 7.0+ — cannot be used on GTX 1080.
# For per-instruction stall analysis, a Volta+ GPU would be needed.
```
