# int3lu Coulomb-fit GEMVs → GPU-resident cuBLAS offload (closed shell)

**Date:** 2026-06-10 · **Status:** DONE (shipped, closed-shell gated) · **Impact:** MED on fosfato-class, HIGH potential on large closed-shell systems

## Problem

`int3lu` (MEMO path) is DRAM-bandwidth-bound: it streams the constant 3-center
integral matrices twice per SCF iteration — GEMV 'N' (`Rc = cool·rho_g`) and
GEMV 'T' (`terms = cool^T·af`), in both fp64 (`cool`) and fp32 (`cools`).

fosfato numbers (Md=804, kknumd=14997, kknums=20707): cool 96 MB + cools 67 MB
→ **326 MB streamed/iter ≈ 27 GB/s** with BLAS=4 inside the overlap section.
On 5800X3D this is most of practical DRAM bandwidth, shared with g2g's CPU
partition. int3lu = **12 ms/iter, the critical path** of the overlapped
section (g2g = 10.3 ms). Threads measured flat (multiZn note) — the bytes are
the wall.

## What shipped

`g2g/cuda/coulomb_fit.cu` + `g2g/coulomb_fit_stub.cpp` (CPU-only builds) +
edits in `subm_int3lu.f90`, `SCF.f90`, `subm_int3mem.f90`, RMMcalc2/4.

- `cool`/`cools` uploaded once per geometry to VRAM; the four GEMVs run via
  cuBLAS (own handle + non-blocking stream, so they interleave with the XC
  kernels). Gather/scatter, `Ginv` DSPMV, energy DDOTs stay on CPU bit-exact:
  **only GEMV summation order changes**.
- **Background prefetch**: upload kicked off at end of `int3mem` (SCF.f90,
  closed-shell guard) on a `std::thread`; iterations use the CPU GEMVs until
  `ready` flips (iter 2 on fosfato). Lessons learned:
  - `cudaHostRegister` on 100+ MB mid-iteration holds the driver lock and
    stalls concurrent XC kernel launches (g2g iter-1 50→84 ms). Pinning was
    **removed entirely** — pageable staged copies contend in short slices.
  - A synchronous upload inside the first overlap section costs ~100 ms wall.
- **Automatic CPU fallbacks** (no env toggles): matrices < 16 MB (TD on small
  systems — PCIe latency would regress chloride), insufficient free VRAM
  (− 256 MB margin; old GPUs / huge systems), any CUDA error, CPU-only build.
- Invalidation: `int3lu_gpu_invalidate()` before every
  `deallocate(cool/cools)` site (int3mem rebuild, SCF finalize, RMMcalc2/4).

## Results (fosfato, RTX 3080 Ti)

- Overlap path: int3lu **12 → 1.1 ms/iter** from iter 2; section now g2g-bound
  (10.3 ms). Wall ≈ break-even at 25 iters (one-time upload ≈ steady savings);
  net win grows with iteration count.
- Default (non-overlap) path: `Coulomb fit + Fock` **~12.5 → 3.5 ms/iter**
  (0.32 s → 0.09 s total) — straight wall win for non-overlap users.
- Energy −2148.6509802 vs −2148.6509796 baseline (0.6 µHa), same 25 iters.
- e2e suite: failure set identical to pre-change baseline
  (08/10/11/12/13 all verified pre-existing by rebuilding the baseline).

## CLOSED-SHELL GATE (and why)

Ungated, heme (open shell) sampled 147/267/227/185 iters vs single baseline
run 100 → looked like a 2× regression. **But**: the gated (CPU-identical)
build then sampled **139/260/155 on the same binary** — heme iteration count
is **run-to-run nondeterministic on hybrid cuda=1 cpu=1 builds** (suspect:
timing-dependent CPU/GPU partition feedback → FP-order shifts). The earlier
OMP-median methodology assumed per-config determinism; on hybrid builds it
does not hold. So the ungated samples were *inconclusive*, and the gate was
kept as the conservative choice.

## UPDATE 2026-06-11: open-shell gate LIFTED

The gate was removed (heme distributions with/without GPU GEMVs overlap given
the hybrid-build nondeterminism; samples 147/267/227/185 ungated vs
139/260/155 gated, same binary class). Two additional pieces shipped:

- **VRAM coordination with the XC cache** (this took two crash iterations to
  get right on multiZn — 5.5 GB cool + 5.47 GB planned XC cache on a 12 GB
  card): the fgm auto-detect runs at SCF *grid setup*, before int3mem, so its
  budget is committed before prefetch. prefetch now (a) reserves the same
  empirical **20%-of-free headroom** regenerate_partition keeps for the
  pool-untracked per-group temporaries (a flat 256 MB margin segfaulted
  `solve_opened`), (b) claims unreserved VRAM first, and (c) carves any
  shortfall out of `GlobalMemoryPool` via `tryAlloc`, so XC just caches fewer
  groups (partial caching) instead of both sides over-committing.
- **Chunked, abortable upload** (256 MB chunks + `abort_upload` atomic):
  SCF teardown no longer blocks behind a multi-GB in-flight copy.

**multiZn (12_Zn_timers input, 20-iter variant, overlap on):** int3lu
**590 → 68 ms/iter** (fully hidden under g2g's 150 ms; section 590→150 ms),
wall **96.1 → 86.9 s (−9.6%)**; 2-iter test: Coulomb fit+Fock 1.65→0.17 s,
XC Fock unharmed (0.55→0.50 s) despite the cache carve-out. ΔRho trajectory
bit-identical for 5 iters, then ulp-seeded divergence (normal for this
chaotic open-shell float32 system).

**Found along the way:** `12_Zn_timers/run.sh` has its liosolo invocation
**commented out** — the suite has been "checking" a stale output file; the
test never runs. Also `os_common.cu` used `cudaGetDeviceProperties`, an
ABI-versioned symbol (plain vs `_v2`) that breaks the liosolo link when
nvcc-12 headers meet libcudart-13 without liohome's library paths; replaced
with `cudaDeviceGetAttribute` (version-stable).

## Follow-ups

1. Characterize/fix the hybrid-build run-to-run nondeterminism —
   it invalidates iter-count-based A/B comparisons for all future work.
2. The 16 MB threshold is a derived constant; revisit on PCIe3 /
   small-VRAM hardware (GTX 1080: same logic holds, slower upload amortizes
   over more iters).
3. Un-comment and fix `12_Zn_timers/run.sh` (test infra).
