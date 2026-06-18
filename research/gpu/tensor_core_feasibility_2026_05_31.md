# Tensor cores / low precision for the LIO GPU path — feasibility study

**Date:** 2026-05-31
**Hardware:** RTX 3080 Ti (GA102, SM 8.6, Ampere), CUDA on PATH 12.0 / alt 13.1
**Workload:** `13_Heme` (open-shell, NUNP=4, 64 atoms, DZVP), `LIO_OVERLAP_INT3LU_G2G=1`
**Status:** STUDY COMPLETE — **negative result with a sharp boundary.** Do not build the
density-as-GEMM tensor-core kernel for this hardware/workload class. Conditions under which
it *would* pay off are listed at the end.

---

## TL;DR

The question was whether Ampere tensor cores (TF32 / FP16 / BF16) can accelerate the
GPU-heavy parts of the LIO SCF. The answer, grounded in two cheap measured probes that
de-risked the expensive kernel rewrite:

1. **Precision wall (decisive).** Emulating TF32 (10-bit mantissa, RNE) in the density
   `ρ = Σ_ij P_ij φ_i φ_j` contraction **roughly doubles heme's SCF iteration count and
   pushes the converged energy up to ~260 µHa off the baseline center, vs a ~19 µHa
   baseline thread-to-thread spread** (≈ 14× wider, decisively out of basin). The iterated
   Fock path is precision-locked; heme's documented Lyapunov sensitivity amplifies raw
   tensor-core precision into divergence. This reconfirms every prior "ulp perturbation
   doubles heme iters" finding, now for the specific case of TF32. The probe was
   **conservative** (only the density-value operands quantized; the live GGA
   gradient/hessian channels left in FP32) and **measured only on heme** as the worst-case
   chaotic transition-metal system — a true full-TF32 GEMM is strictly worse.

2. **Performance ceiling (even if precision were free).** At heme's *actual* GEMM shapes
   (contraction dim K = group_m ≈ 128–256), cuBLAS tensor cores deliver only **1.2–1.5×
   (TF32) / 1.6–1.9× (FP16)** over SGEMM — nowhere near the 2×/4× spec, because small K
   starves the MMA pipeline. That GEMM is additionally **capped at ~5 % of heme wall** by
   the int3lu/g2g overlap, and the density-as-GEMM reform would *add* W-materialization
   global-memory traffic that the current register-resident fused kernel avoids.

The only precision-safe tensor-core path is **error-compensated emulation** (3×TF32 /
Ozaki ≈ 3 GEMMs to recover FP32 accuracy). At a 1.2–1.5× raw TF32 advantage, 3 GEMMs net
**< 1×** — a loss. **Verdict: tensor cores are not a wall-time lever for the heme-class
SCF on this or older hardware.**

---

## Where GPU time actually goes on heme

Baseline run (72 iters at default threads, 17.5 s total). Per-iteration `[overlap]` line:
`int3lu = 82 ms, g2g = 93.6 ms, idle = 11.6 ms`. Under `LIO_OVERLAP_INT3LU_G2G=1` the
"Fock integrals" bucket = max(int3lu, g2g) per iter, and **g2g (the GPU XC pipeline) is
the binding leg** by only ~11.6 ms.

| Iteration sub-cost | time | where |
|---|---|---|
| Fock integrals (= max(int3lu, GPU-XC)) | 6.85 s (45 %) | GPU XC binds, int3lu hidden under it |
| SCF Fock diagonalization | 3.74 s (24 %) | CPU LAPACK (dsyev*) |
| SCF acceleration setup (DIIS) | 3.22 s (21 %) | CPU BLAS (BChange/commut/update_emat) |
| DIIS solve | 0.49 s (3 %) | CPU |

Two consequences:
- The GPU XC pipeline (≈ density kernel, 85 % of GPU time per prior profiling) is the
  single biggest contributor — the right place to look for a GPU lever.
- **But the overlap caps its wall payoff:** even an infinitely fast GPU density kernel
  drops Fock-integrals only to the 82 ms int3lu floor → `(93.6−82)×72 ≈ 0.84 s ≈ 4.8 %`
  of total. Heme's CPU/GPU balance pre-caps any density-kernel win.

### ncu on the density kernel (`gpu_compute_density_opened<float,false,*>`)

The launches split sharply by group size:
- **Small launches** (group_m ≤ 64, ~95 blocks × 64 threads): **Achieved occupancy ~5 %,
  Compute (SM) ~12 %, DRAM ~1 %, ~9 µs each.** Latency/occupancy-starved — *not*
  compute-bound, so tensor cores cannot help them; they are launch-overhead limited.
- **Big launches** (group_m > 128): prior profiling (memory) shows ~48 % occupancy /
  ~72 % SM throughput, register-pressure limited at ~24 active warps/SM. These *are*
  FMA-pipe bound, hence the only tensor-core-amenable population.

So tensor cores could in principle help only the big-group density launches — a subset of
the 85 %, itself capped at ~5 % wall.

---

## The density kernel is structurally a GEMM

`g2g/cuda/kernels/energy_open.h` (and the identical closed-shell `energy.h`): per point r,
per basis row i, the hot loop accumulates

```
w_a[i]  += P_a[i,j] · φ_j(r)          (+ gradient ×3, hessian ×6 reuse of P_a[i,j])
ρ_a(r)   = φ_i(r) · w_a[i]
```

i.e. `W = P · Φ_ext` where `Φ_ext = [φ | ∂x | ∂y | ∂z | h1(3) | h2(3)]` (10 components),
followed by a cheap element-wise reduction `ρ = Σ_i φ_i W_i`. That `P·Φ` is a genuine
GEMM: `M = N = group_m` (≈ K, the contraction), `N_cols = 10·npoints` (wide). It maps
cleanly onto tensor cores *if* precision allows — which is exactly what was tested.

---

## Probe 1 — TF32 precision in the density contraction (DECISIVE)

**Method.** Added a compile-time-gated `tf32_emulate()` (round FP32→TF32, 10-bit mantissa,
RNE: `(u + 0x1000) & 0xFFFFE000`) and wrapped both multiply operands of the density
contraction — the P-matrix texture fetches *and* the basis value `fj` — so every product
becomes a faithful TF32×TF32→FP32 MAC, exactly as an Ampere TF32 GEMM would compute it.
No GEMM written; the existing kernel's arithmetic was quantized in place. Reverted after.

**Verdict metric (per advisor):** final energy vs the −2829.6326 ±1 µHa basin, *and* iter
count across OMP={1,4,6,8} — because heme iter count is Lyapunov-noisy and a single run
cannot distinguish "broke convergence" from chaos.

| | iters (OMP 1 / 4 / 6 / 8) | energy (OMP 1 / 4 / 6 / 8) |
|---|---|---|
| **Baseline FP32** | 134 / 83 / 185 / 99 | −2829.63261 / .63262 / .63261 / .63260 |
| **TF32 density**  | 158 / 329 / 319 / 174 | −2829.63248 / .63265 / .63235 / .63235 |

- Baseline energy spread across threads: **~19 µHa** (min −2829.6326159, max
  −2829.6325970; center ≈ −2829.632607) — tight, in basin.
- TF32 energy spread: **~300 µHa**, biased high; the worst run (−2829.6323480) is
  **~260 µHa above the baseline center**, and three of four runs land 130–260 µHa off
  → TF32 converges to a *perturbed DFT state*, not just slower (~14× the baseline spread).
- TF32 iteration count: median roughly **doubles** (≈245 vs ≈115).

**Conservatism of the probe.** Only the density-value multiply operands were quantized
(`fjreg` and the four `rdm` texture fetches). Heme runs GGA (`lda=false`), so the
gradient/hessian basis channels (`fgjreg`/`fh1jreg`/`fh2jreg`) are live but were left in
FP32; a real full-TF32 density GEMM would quantize those too. The probe therefore
*understates* the error a true reformulation injects — and it already broke convergence.

**Conclusion:** raw TF32 in the density/Fock path is non-viable for heme. The 10-bit
mantissa injects ~1e-3 relative error into ρ → XC potential → Fock; the SCF fixed point
both slows and shifts. (BF16, 8-bit mantissa, would be strictly worse; FP16 lacks the
exponent range for basis values regardless.)

---

## Probe 2 — Tensor-core GEMM ceiling at heme shapes (cuBLAS microbench)

`research/gpu/tc_gemm_probe.cu` — `cublasGemmEx` FP32 vs `COMPUTE_32F_FAST_TF32` vs
`COMPUTE_32F_FAST_16F`, at `C[M,N] = Pᵀ[M,K]·Φ[K,N]` with K = group_m:

```
shape                          FP32(ms)   TF32(ms)   FP16(ms)    TF32x  FP16x
small group, ncols=4096          0.0149     0.0097     0.0081    1.53x  1.83x
mid group,   ncols=8192          0.0463     0.0397     0.0287    1.17x  1.62x
large group, ncols=16384         0.1205     0.1006     0.0637    1.20x  1.89x
big basis,   ncols=16384         0.4224     0.3054     0.2364    1.38x  1.79x
```

**Small K (= group_m) is the killer.** Tensor cores want K large to amortize the MMA
pipeline; at K = 128–256 the realized speedup is **1.2–1.5× (TF32) / 1.6–1.9× (FP16)**,
not the 2×/4× peak. And this is the GEMM *in isolation*: the density-as-GEMM reform must
also materialize `W` (group_m × 10·npoints × 2 spins) to global memory — traffic the
current fused kernel avoids by keeping `w` in registers — which on small-K (memory-leaning)
GEMMs further erodes the gain.

**Net for a precision-safe path (projection, not measured):** error compensation
(3×TF32 ≈ FP32 accuracy) costs ~3 GEMMs. From the measured 1.2–1.5× TF32 advantage,
3 × (1/1.5) ≈ **2× slower** than the single SGEMM it replaces. Even FP16-based 2-pass
schemes project to ≤ 1× here. There is no positive-EV precision-safe tensor-core GEMM at
these shapes. (This is arithmetic off Probe 2's measured ratios; the compensated kernel
itself was not built — the precision and ceiling probes already ruled it out.)

---

## Why the rest of the heme hot path is also tensor-core-hostile

- **Diagonalization (3.74 s, CPU dsyev*)**: needs FP64; GA102 has **no FP64 tensor cores**
  (1:64 FP64 rate). cuSOLVER was already tried and rejected (slower at M=364 *and* perturbs
  the near-degenerate Fe-d basis → doubles iters). FP64 emulation via Ozaki/INT8 would only
  beat native FP64 — irrelevant here since LIO diagonalizes on CPU in true FP64.
- **DIIS setup (3.22 s, CPU BLAS GEMMs C=CᵀFC etc.)**: documented as ferociously
  ulp-sensitive (DGELS rank-deficiency: any ulp shift flips the LS pick and slows the
  trajectory). TF32 here = guaranteed basin/iter damage, same mechanism as Probe 1.
- **update_rmm (XC-Fock GEMM, already cuBLAS SGEMM)**: untested in TF32, but it feeds the
  *same* Fock matrix density does, so the same basin-shift risk applies; and at ~4 % of the
  Fock (Coulomb=7284 vs XC=−309 Eh) and already fast, the upside is negligible. Not worth a
  probe.

---

## When tensor cores *would* be worth it (forward-looking boundary)

The negative result is specific to **small, chaotic, CPU-balanced, overlap-capped** heme.
The lever turns positive when *all* of these hold:

1. **Large contraction dim** — group_m / basis-per-group ≳ 256–512 (large molecules, big
   basis), so the GEMM approaches tensor-core peak (≥ ~2×, see "big basis" row trending up).
2. **GPU XC genuinely on the critical path** — a single-point or forces evaluation, or a
   system where int3lu does *not* shadow g2g, so a density speedup is not pre-capped at 5 %.
3. **Benign convergence** — closed-shell, non-transition-metal, no near-degenerate frontier
   orbitals, so error-compensated (3×TF32 / 2-pass FP16, ≈ FP32 accuracy) precision is
   tolerated by DIIS. Plain TF32 stays off the table even there.
4. **Non-iterated targets preferred** — post-SCF properties, real-space TD-DFT propagation
   with a stable propagator, ML-inference surrogates: one-shot evaluations where there is no
   fixed-point feedback to amplify precision error. (LIO TD is currently host-bound, so not
   today's bottleneck.)

A concrete *next* experiment, if pursued: build the density-as-GEMM with **FP16 + 2-pass
error compensation**, gate it `#if __CUDA_ARCH__ >= 800` with the current FP32 fused kernel
as the Pascal/Turing-and-below fallback (GTX 1080 has no tensor cores — an explicit CLAUDE.md
target), and evaluate on a **large closed-shell single-point** (not heme). Only proceed if a
roofline there shows the big-group GEMM is FLOP-bound *after* accounting for W traffic.

---

## Artifacts

- `research/gpu/tc_gemm_probe.cu` — tensor-core GEMM ceiling microbench (kept).
- TF32 density probe — implemented in `energy_open.h` behind `-DTF32_DENSITY_PROBE`,
  measured, **reverted** (tree is clean; reproduce by re-applying the `tf32_emulate` wrap
  on the two operands of `w_a/w_b += rdm*fjreg`).
- `test/LIO_test/13_Heme/sweep_omp.sh` — OMP={1,4,6,8} energy+iter harness for the chaos-band
  verdict.
