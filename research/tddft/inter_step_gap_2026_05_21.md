---
name: td-inter-step-gap
description: TDDFT open-shell propagation has a large GPU idle pool between consecutive TD steps; identifies the structure and possible attack vectors.
status: PARTIAL — vector 4 measured DEAD; magnus host-buffer caching landed (~2.9% wall on 50k chloride.in)
impact: medium (~1.2s out of 41.5s wall on chloride.in 50000 steps after fix)
metadata:
  type: project
  workload: 07_TDDFTHCL/chloride.in (open shell, propagator=2, ntdstep=50000)
  date: 2026-05-21
  hardware: RTX 3080 Ti (Ampere SM 8.6), CUDA 12.0
---

# TDDFT open-shell — inter-step GPU idle gap

## Profile evidence

`LIO_OVERLAP_INT3LU_G2G=1 nsys profile … liosolo -i chloride.in -c chloride.xyz -v`

| Metric | Value |
|---|---|
| Wall | 4.67s (matches LIO timer `TD - TD Step: 4.62s`) |
| Total GPU kernel time | 2577ms |
| GPU busy fraction | **55.3%** |
| Total kernel launches | 306,366 |
| Idle gaps **5–50µs** (inter-group within a step) | 40,426 gaps, **368ms** total |
| Idle gaps **50–500µs** (inter-step) | 5,240 gaps, **~1300ms** total |
| Idle gaps **> 500µs** (setup/finalize) | 49 gaps, ~800ms |

## What the 1.3s inter-step pool actually is

Looking at a 5ms mid-run window, one TD step has this shape:

```
~570µs of dense GPU work (8 groups × {gather, density, accumulate, dgmm, sgemm, splitK, scatter})
~230µs CPU-only gap   ← inter-step
~570µs next step
~230µs CPU-only gap
…
```

The 230µs gap × ~5000 steps ≈ 1.15s, plus ~150ms of larger Magnus-related gaps = the 1.3s pool.

During those 230µs the CPU is running the propagator (Magnus second-order, `propagator=2`) on the host: forming the new density matrix from the just-computed Fock, applying the basis change, and feeding the next step's P matrix back into Fortran-side state.

## Why this matters

- It is the **single biggest pool of latent improvement** in TD chloride open-shell.
- Per-step CPU work that gates GPU launch is independent of the GPU's TFLOPS — buying a faster GPU does not shrink this.
- Anything in the g2g side (multi-stream, kernel fusion, graphs) cannot touch this because the GPU has no queued work during the gap.

## Attack vectors (not yet evaluated)

### 1. Move the Magnus propagator step to the GPU
Currently the propagator runs in Fortran on the host (lioamber/ehrensubs/). The matrices are small (basis size ~80 for chloride), but cuBLAS GEMM + matrix-exponential on device would push the gap to ~0. Risk: TD propagator is numerically sensitive; moving to float32 GEMMs would change energies. FP64 GEMM on 3080 Ti is 1:32 throughput — but matrices are tiny (~80×80) so latency dominates, not throughput, and host↔device round trip per step may be net loss.

### 2. Overlap propagator with the next step's *density-only* GPU work
The propagator needs the **Fock matrix** that the current step produced. But the *density-on-grid* part of the next step only needs the propagated P matrix, which the propagator is computing. There's no overlap unless the pipeline is restructured. **Likely dead end** without restructuring.

### 3. Pipeline: speculatively start step N+1's density evaluation
Speculative: predict the next P (e.g. linear extrapolation from previous steps), launch density-on-grid in parallel with the propagator, then if the prediction was wrong (it always is, formally) re-launch. Not useful — would always re-launch.

### 4. Parallel propagator on host CPU threads
The propagator is sequential per-step but uses BLAS internally. If BLAS is single-threaded (current `LIO_OVERLAP_BLAS_THREADS` setting throttles it), increasing threads during the inter-step gap could shorten the 230µs gap directly. Lowest-risk first attempt: time `td_calc_energy` / `predictor` on host with `OMP_NUM_THREADS` cranked up during the gap only. The 5800X3D has 8 physical cores — plenty of slack.

### 5. Fold the Fock build into the propagator step
If int3lu (Coulomb fit + Fock) can be partially overlapped with the propagator on the same step that finished, the visible inter-step gap shrinks. The existing `LIO_OVERLAP_INT3LU_G2G=1` already overlaps int3lu with g2g *within* a step; the open question is whether anything in the propagator's serial portion can move into the g2g window.

## Measurement update (2026-05-21, this session)

Instrumented `magnus()` and `predictor()` in `lioamber/propagators.f90` to break the
gap into discrete phases on `chloride_short.in` (150 steps, open-shell).

### Build context (was missing from original analysis)
The default build is `make cuda=1`, which does **NOT** define `-DCUBLAS`. With
`-DCUBLAS` undefined, `cumat_x` in `lioamber/typedef_cumat/` falls back to host
arrays + host BLAS (zgemm/zaxpy). The magnus BCH loop is therefore 100% on the
CPU in this build — the "host-side propagator" framing of the original analysis
was correct, but the reason is the build flag, not an architectural decision.

### Per-call breakdown on HCl (M=21, NBCH=10)
- `alloc(Omega1)` + 4× `cumat_x%init()`: ~0.6 µs
- BCH loop (10 iters × {2 ZGEMM + 1 ZAXPY + 1 exchange}): ~28 µs
  - ZGEMM(21,21,21) × 2: ~1.2 µs each (2.4 µs/iter; 84% of loop)
  - ZAXPY: ~100 ns/iter
  - **exchange via alloc/copy/copy/copy/dealloc: ~350 ns/iter**
- `%get` + 4× `%destroy` + `dealloc(Omega1)`: ~0.3 µs
- **Total magnus call: ~30 µs**

Per Magnus-regime TD step (open shell): 4 magnus calls = ~120 µs CPU-only.
Per-step propagation total ≈ 770 µs; magnus is ~16% of that, ~70% of the
purely CPU-only "gap" portion.

### Vector 4 (BLAS thread cranking) — DEAD
Tested explicitly via `LIO_MAGNUS_BLAS_THREADS={1, 6}` (default 6 from g2g
auto-tune). BCH loop time: **27.16 µs at 6 threads, 27.22 µs at 1 thread**.
For 21×21 ZGEMMs OpenBLAS dispatch is already at the floor — thread count
is irrelevant. The 4-thread "throttle" in `SCF.f90` (`LIO_OVERLAP_BLAS_THREADS`)
does not apply to the TD path; nothing was capped here to crank.

**Implication**: do not retry vector 4 on this code path. It's not a lever.

### Vector 1 (GPU propagator) — context update
Vector 1 is "move magnus to GPU." The `cumat_x` already has the CUBLAS path;
`make cuda=2` enables it. For HCl (M=21), each ZGEMM(21,21,21) is ~74 KFlops,
~1 µs of FP work. cuBLAS launch overhead is ~5-10 µs per kernel; 80 launches
per magnus call would balloon to ~400-800 µs instead of the current 28 µs.
**Vector 1 is dead for small molecules on the existing per-ZGEMM dispatch.**
It would need a batched/graph approach to be net-positive at this size.

## Implemented in this session

Two changes landed (`lioamber/propagators.f90`,
`lioamber/typedef_cumat/cumat_exchange.f90`):

1. **Cache magnus working buffers across TD steps** (SAVE'd `cumat_x` +
   Omega1). Replaces per-call alloc/destroy of 4× M×M complex matrices with
   one-shot lazy allocation, sized on first call.
2. **`exchange_x`/`exchange_r` via `move_alloc`** (non-CUBLAS path). Replaces
   `allocate(tmp) → 3× array copy → deallocate(tmp)` with three O(1)
   descriptor swaps. Called 10× per magnus call (BCH loop) and was ~350 ns
   each; now negligible.

### Measured impact on chloride.in (50000 steps, open-shell, propagator=2)

| Metric | Baseline | After | Δ |
|---|---|---|---|
| Total time | 41.527 s | 40.322 s | **−1.205 s (−2.9%)** |
| TD - TD Step | 41.417 s | 40.216 s | −1.201 s |
| TD - Propagation | 39.470 s | 38.630 s | −0.840 s |
| wall (`time real`) | 41.819 s | 40.623 s | −1.196 s (−2.9%) |

Bit-identical dipole moment on `chloride_short.in` vs baseline (1.5100137...).

### What I tried that regressed

Direct-BLAS bypass of `cumat_x` in the BCH loop (raw `xgemm`/`xaxpy` + raw
`move_alloc` swap): TD-Propagation went from 106 ms → 110 ms on
`chloride_short.in`. Fortran type-bound procedure dispatch on a concrete
type is essentially static; the cumat_x wrapper overhead is below noise.
Avoid this restructure — it adds an #ifdef CUBLAS fork with no win.

## What's left in the inter-step gap

After the buffer-caching fix, magnus is still ~25 µs/call × 4 calls/step ≈
100 µs/step of pure BCH-loop time. The 84% of that loop is genuine BLAS
work (2.4 µs/iter × 10 iters in ZGEMMs). Further reduction requires either:

- **Algorithmic change**: precompute `U = exp(-i F dt)` via Padé(6,6) once
  per magnus call, then `ρ_new = U ρ U†` (2 ZGEMMs vs 20). Same numerical
  series, different truncation form; needs correctness gate (TD uses FP64
  internally so DIIS-style float32 sensitivity is not a blocker, but a
  long-run dipole-moment comparison is required).
- **Custom fused commutator kernel**: replace 2× ZGEMM with one M³-pass
  computing `α (BA − AB)`. ~2× faster theoretically; ugly and specialized.

Neither is in scope of this fix.

## Cross-references

- [[overlap_autotune_2026_05_03]] — autotune for OMP / BLAS thread counts on the SCF overlap path; same dial would govern TD propagator
- [[td_overlap_int3lu_reverted_2026_05_19]] — earlier per-TD-step int3lu overlap was reverted because per-step OMP team spawn dwarfed the savings. **Important constraint**: any per-step overlap mechanism must avoid spawning OpenMP teams per step.
