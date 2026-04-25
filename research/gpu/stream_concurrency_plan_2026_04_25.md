---
status: PLAN
date: 2026-04-25
impact: 200-500 ms estimated (10-25% wall) — best remaining lever
---

# Stream concurrency / GPU occupancy plan

## TL;DR

The GPU is **idle 65% of wall time** on fosfatoQMMM (1.36 s of stream-7 gaps
out of 2.09 s total). The idle is *not* inside g2g — it's *between* the 25 SCF
iterations, while Fortran does int3lu / diag / DIIS on the CPU and the GPU
queue is empty.

The previously-rejected "Multi-stream GPU pipeline" research is correct
**within g2g** (single-iter GPU is well-utilized). The opportunity is one
level up: the SCF main loop is strictly sequential and forces the GPU to a
full stop between iterations. Stream parallelism that crosses the
iteration boundary, plus CUDA Graphs to amortize launch overhead, is the
big remaining lever.

## Evidence

From `nsys` profile (post-async-pool, fosfatoQMMM, RTX 3080 Ti):

| Stream | Kernels | Total kernel time |
|---|---|---|
| 7 (default, all SCF + most AINT) | **5944** | **494 ms** |
| All others (transpose, AINT post-SCF) | ~12 | ~46 ms |

**Stream 7 idle time** (gap between consecutive kernels):

| Gap bucket | Count | Total |
|---|---|---|
| **> 10 ms** | **30** | **1358 ms** |
| 1–10 ms | 4 | 6 ms |
| 100 µs – 1 ms | 95 | 19 ms |
| 10 – 100 µs | 309 | 12 ms |
| < 10 µs | 5505 | 14 ms |

**Top 5 gaps**: 384, 105, 58, 47, 38 ms. The 384 ms gap is post-SCF
phase setup; the 25 gaps in the 30-40 ms range are inter-iteration
Fortran work (diag + DIIS + int3lu + Fock matrix bookkeeping).

**Wall time decomposition (2.09 s):**
- GPU kernels (real work): ~525 ms
- GPU idle waiting for CPU: ~1360 ms (65%)
- CPU-GPU API overhead, syncs, etc.: ~200 ms

## Where the gaps come from (per iter, ~47 ms)

```
[g2g_solve dispatched]
  ├─ ~16 ms GPU kernels (density + rmm_update + accumulate × 44 groups)
  ├─ ~13 ms CPU partition (concurrent with GPU)
  └─ small CPU finalize

[g2g_solve returns]
  ├─ int3lu               ~10.7 ms CPU only — GPU IDLE
  ├─ Fock assembly         ~2 ms   CPU only — GPU IDLE
  ├─ Diagonalization      ~10 ms   CPU only — GPU IDLE
  ├─ DIIS extrapolation    ~5 ms   CPU only — GPU IDLE
  └─ rho update            ~1 ms   CPU only — GPU IDLE
                       ──────────
                       ~30 ms GPU idle/iter
                       × 25 iters = 750 ms wasted GPU time
```

The numbers above match the 30 large gaps × ~30 ms = ~900 ms observed.

## Plan, ranked by expected payoff

### Tier 1 — Cross-iteration overlap (Fortran-side, biggest lever)

**Estimated impact: 200-300 ms (10-15% wall).**

**1.1. int3lu ↔ g2g_solve overlap** *(already research item #1,
[fortran/overlap_int3lu_g2g.md](../fortran/overlap_int3lu_g2g.md))*

`int3lu` (Coulomb J on CPU) and `g2g_solve` (XC on GPU+CPU) are
algorithmically independent — they read the same density and produce two
separate Fock contributions that get summed. Currently strictly serial.
Run `int3lu` on a CPU thread concurrent with `g2g_solve` calling its GPU
work: 10.7 ms × 25 = 268 ms ceiling, realistic 150-220 ms after sync
overhead.

**1.2. Diagonalization on GPU** (or CUDA-side eigensolver in cuSOLVER)

10 ms × 25 = 250 ms idle. cuSOLVER's `cusolverDnDsyevd` runs on GPU. Even
if not faster than CPU MKL, running it on a stream parallel to next-iter
Fock build (when ready) buys overlap. **Caveat**: currently nothing useful
runs in parallel because next iter needs the eigenvectors; would only help
if combined with double-buffered SCF.

**1.3. Pre-issue next-iter Fock GPU work while current iter does DIIS**

DIIS (~5 ms) updates the density predictor for the next iter. But the
**current** iter's Fock GPU dependencies (RMM upload from current density)
can be staged before DIIS finishes if we accept a one-iter pipeline delay.
Complex; defer to after 1.1.

### Tier 2 — CUDA Graphs to eliminate launch overhead

**Estimated impact: 30-60 ms (1.5-3% wall). Cheap to implement.**

5505 sub-10µs gaps total 14 ms — these are largely launch latency.
Plus 309 mid-range gaps total 12 ms. cudaLaunchKernel API time is 48 ms
(7.9 µs × 6112 launches).

**2.1. Capture per-iter SCF kernel sequence as a CUDA graph.**
Each SCF iter launches ~244 kernels in a fixed pattern (gather → texture
upload → density → accumulate → rmm_update → scatter, repeated for 44
groups). Capture once on iter 1, replay for iters 2-25.

Implementation:
1. Detect "stable" iter (skip iter 1 setup churn, use fgm=-1 caching).
2. `cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)` around
   the per-group kernel launches in `Partition::solve_groups` (only the
   GPU-thread bin's groups).
3. `cudaStreamEndCapture` → `cudaGraph_t`. Instantiate to
   `cudaGraphExec_t` once.
4. Subsequent iters: `cudaGraphLaunch(exec, stream)` instead of the
   per-kernel launch loop.

Risk: graph parameters (pointer offsets, sizes) must be invariant across
iters. With `fgm=-1` caching this is largely true after iter 1.
Conditional/dynamic grids (e.g., based on group n_indexes) freeze. If a
group shape changes between iters (rebalancer reassigns groups to bins),
graph must be re-captured. Mitigation: re-capture if `rebalance()`
changed bin assignment.

**2.2. Replace per-iter `cudaMemcpyAsync(rmm)` and `cudaMemset(fock)` with
graph-captured nodes.**

### Tier 3 — Within-g2g pipelining (smaller wins)

**Estimated impact: 15-30 ms (1% wall).**

**3.1. Per-group stream pipelining via N persistent streams.**
Currently 44 groups × {gather, tex-upload, density, accumulate, rmm,
scatter} on stream 0. Use 2-4 round-robin streams so group N+1's gather +
texture upload (the *copy-engine-feasible* parts) overlap with group N's
density (SM-bound).

Ampere has separate copy engines, so the texture upload (~13 µs / group ×
44 groups × 25 iters = 14 ms theoretical max) can fully overlap density
on a different stream. Need cross-stream synchronization for the global
Fock atomicAdd at scatter time.

Hazards:
- `gpu_scatter_rmm` atomicAdd on shared `s_global_fock_dev` is fine
  device-wide, but ordering with the final D2H Fock readback requires
  cudaEvent on each stream → cudaStreamWaitEvent before sync.
- Texture array (`rmm_cuArray`) is per-group, no contention.
- `factors_gpu`, `partial_densities_gpu` etc. are per-group → safe.

**3.2. Dispatch density / rmm_update on different streams within a group.**
*Rejected*: density produces `factors_gpu` which rmm_update consumes.
True data dependency.

**3.3. Pre-stage next-iter rmm upload during current-iter scatter.**
Once current iter's scatter is queued, the host knows the next iter's
RMM read won't start until D2H Fock readback completes. If we
double-buffer `s_global_rmm_dev` and overlap H2D with current scatter,
save ~half a millisecond per iter. Marginal.

### Tier 4 — Multi-host-thread GPU dispatch (rejected re-eval)

**Estimated impact: minimal on this benchmark.**

Current rejection (`stream_sharding.md`) holds: a single density kernel
already saturates Ampere's 84 SMs (219 µs / call, 1188 calls = nearly
continuous SM occupancy when launched). Adding more host threads each
issuing density on different streams would just queue them sequentially
on the SMs.

**Exception**: scenarios where many small groups exist (small molecule
or after rebalancing). Then concurrent execution of multiple small
kernels can co-resident. Worth trying once Tier 2 is in place — graphs
naturally express the parallelism.

## Independent: AINT post-SCF concurrency

The 384 ms top gap is the AINT setup phase. AINT already uses
NUM_TERM_TYPES (6) streams in `qmmm.cu` for fock/forces by angular
momentum tier. The *setup* before those launches is sequential and CPU-
bound. Profile to see what's spending 384 ms — likely
`coulomb_aint_init` or `qmmm_aint_init`.

## Ordered implementation roadmap

1. **Tier 1.1 (int3lu ↔ g2g)** — biggest single lever, 150-220 ms,
   already has a research doc. Fortran + threading.
2. **Tier 2.1 (CUDA graph)** — 30-60 ms, self-contained in g2g. Lower
   risk than Fortran restructuring.
3. **Tier 3.1 (per-group streams)** — 10-15 ms, pure g2g change.
4. Profile AINT 384 ms gap (before optimizing it; may already be I/O-
   or alloc-bound).

## What NOT to do (rejected here)

- **Custom GPU memory arena (Tier 3 of allocator plan)** — superseded by
  `cudaMallocAsync` (committed 2026-04-24).
- **`__ldg` / texture removal** — proven 2.5× regression on Ampere.
- **Replicating density across streams** — single density already
  SM-saturated.
- **Aufbau initial guess / level shifting** — proven not a lever.

## Measurement methodology

- Reference benchmark: `test/LIO_test/03_fosfatoQMMM/run.sh` (verbose=0,
  mulliken+dipole+forces enabled).
- 10-run median wall, hot disk cache.
- nsys profile sqlite query for stream-7 gap analysis (see this doc's
  Evidence section).
- Correctness gate: full `./run_tests.py` (30 tests). Open-shell
  Fe3H2O6 must NOT regress (caught the 2026-04-25 cudaHostRegister bug).
