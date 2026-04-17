# int3lu GPU Offload — Evaluation

**Status:** REJECTED — dominated by CPU/GPU overlap (same ceiling, 3-5× more work)
**Date:** 2026-04-17

## Premise

`int3lu` (Coulomb fit + Fock assembly) is the largest remaining CPU cost in
the SCF iteration: 10.7ms/iter × 25 = 268ms (15% wall). It runs sequentially
with `g2g_solve_groups` (~20ms/iter GPU+CPU partition). The question:
*instead of* overlapping int3lu with g2g on CPU/GPU, port int3lu itself onto
the GPU via cuBLAS + custom scatter kernel.

## Int3lu structure

From `lioamber/faint_cpu/subm_int3lu.f90`:

1. **Density projection** (DGEMV): af[k] = Σᵢⱼ cool[k, ij] · Pmat[ij]
2. **Fit coefficient solve** (DSPMV): af' = Ginv · af   (Md × Md packed symm)
3. **Fock accumulation** (scatter): Fmat[kkind[kk]] += Σₖ af'[k] · cool[k, kk_ind]
4. **Single-precision variant** of (1)+(3) via `cools/kkinds/sgemv` for
   screened/approximate terms

Typical sizes (fosfato, Md=804): `cool` array is O(Md × packed_pairs) ≈
tens of MB; Pmat/Fmat packed MM=323k doubles.

## GPU offload design (hypothetical)

| Step | GPU approach | Size | Notes |
|------|--------------|------|-------|
| Mirror cool/cools | one-time H2D after int3mem | tens of MB | memory footprint OK on 12GB RTX 3080 Ti |
| Ginv residency | once per SCF init | 804² = 5MB | trivial |
| DGEMV density projection | cuBLAS `cublasDgemv` | 804 × ~15K | small for GPU; 2-3× CPU best case |
| DSPMV fit solve | cuBLAS `cublasDspmv` | 804 packed | launch-overhead bound |
| Fock scatter accumulation | **custom kernel** | kkind indirection | 1-2 weeks of kernel dev |
| Pmat H2D / Fmat D2H per iter | cudaMemcpyAsync | 2 × 323K doubles = 5MB/iter | ~50µs each with pinned mem |

## Ceiling analysis

Two scenarios for scheduling int3lu-GPU vs. g2g_solve:

**Scenario A: sequential (int3lu-GPU runs, then g2g_solve)**
- Critical path per iter: 3ms + 20ms = 23ms
- Save vs current (30.7ms): 7.7ms/iter × 25 = **192ms**
- **Less** than pure CPU/GPU overlap (267ms ceiling)

**Scenario B: concurrent on separate streams**
- int3lu-GPU + g2g density compete for 80 SMs
- Density kernel is at ~52-69% of roofline (per `gpu/roofline_gpu_compute_density.md`),
  so there are spare SMs for small cuBLAS work
- Realistic: critical path drops to ~20-21ms vs overlap's 20ms
- **Save vs current: ~242ms = tied with overlap**, within noise

## Why it loses vs. CPU/GPU overlap

Both approaches have the same *ceiling* (~267ms). The difference is cost:

| Dimension | int3lu GPU offload | CPU/GPU overlap |
|-----------|-------------------|-----------------|
| Lines of new C++/CUDA code | 500-800 (scatter kernel + memory mgmt + cuBLAS glue) | ~50 (OpenMP sections + buffer split) |
| New failure modes | float32 scatter non-determinism, device OOM on large Md, CUDA memcheck/racecheck | OpenBLAS/libgomp thread contention (tuneable) |
| Validation scope | Full e2e + unit tests for new scatter kernel | Full e2e (same as overlap) |
| Reversibility | Large — tied into device memory lifecycle | Small — env var toggle |
| Implementation | 3-4 weeks | 1-2 weeks |

**The only condition under which offload dominates overlap is if the GPU has
so much spare capacity during g2g_solve that int3lu fits "for free" in
parallel streams — i.e. the entire 10.7ms disappears. On a 20-SM GPU running
a compute-kernel that uses 40-70% of SMs, this assumption doesn't hold.**

## Combined approach (overlap + partial offload)

Running int3lu in its own OpenMP section *and* offloading its DGEMV to cuBLAS
is an option. The incremental gain over pure overlap is small (+5-10% of the
overlap savings) and the complexity is additive: both thread-management
issues *and* device-memory issues *and* scatter-kernel correctness. Not
worth it until the simpler overlap has landed and been measured.

## Conclusion

Use CPU/GPU overlap (see `overlap_int3lu_g2g.md`). If after overlap the
per-iter critical path is still dominated by int3lu's CPU work (it won't be
for fosfato-sized systems, but might be for larger M), revisit this document.

## When to revisit

- If system size grows to M ≥ 1500, cuBLAS DGEMV becomes clearly dominant
  over OpenBLAS; offload ceiling grows.
- If a future LIO version uses a fully GPU-resident density matrix (currently
  Pmat lives on CPU), H2D transfer overhead vanishes.
- If int3mem's `cool/cools` arrays are themselves moved to GPU memory (for
  reasons independent of int3lu), the memory-mirroring cost drops to zero.
