---
status: REJECTED
date: 2026-04-25
impact: NEGATIVE (~2.5× slower than plain FULL_DOUBLE; zero accuracy gain)
risk: realised
area: g2g/cuda/kernels
---

# FULL_DOUBLE + Kahan in density/RMM kernels — dead end

## Hypothesis

After confirming the FULL_DOUBLE build works correctly post the 2026-04-19
shared-memory race fix (see `reproducibility_investigation_2026_04_19.md`),
the natural follow-up was: **add Kahan compensated summation to the FP64
GPU kernels**. The original Kahan rejection (2026-03) was driven by float32
DIIS sensitivity — with FP64 precision headroom, that constraint is gone,
so Kahan should be safe to evaluate on its merits.

Two questions:
- **a.** Does Kahan in FP64 reduce SCF iter count or improve numerical
  stability beyond what FP64 alone provides?
- **b.** Does it introduce any new bug?

## What was tried

`g2g/cuda/kernels/energy.h` `gpu_compute_density` bj-loop (lines 142-184):
8 accumulators (`w, w2, w3, ww1, ww2, w32, ww12, ww22`) wrapped with
`kahanAdd`/`kahanAdd3` from the existing `kahan.h`. Same change in
`g2g/cuda/kernels/rmm.h` `gpu_update_rmm` inner loop (`rmm_local`).

Build: `cuda=1 cpu=1 precision=1`. Test: fosfatoQMMM, 3 runs.

## Results

| Metric | Hybrid baseline | FULL_DOUBLE | FULL_DOUBLE + Kahan |
|---|---|---|---|
| Iterations | 25 | 27 | **27 (no change)** |
| Final energy | −2148.6509632 | −2148.6509303 | **−2148.6509303** (bit-identical to FD) |
| Wall time | 2.24 s | 5.7 s | **12.3–17.9 s** |
| Run-to-run noise | ~200 µHa (rebalancer) | 0.1 µHa | 0.2 µHa |

## Why it doesn't help (a)

Numerically: with FP64 ε ≈ 2e-16, summing ~M terms (M=86 for fosfato) of
O(1) magnitude gives worst-case sum error O(M·ε) ≈ 2e-14. The SCF
convergence threshold is 1e-6, so the FP64 sum has **~8 orders of
magnitude of headroom**. Kahan compresses 2e-14 → ε ≈ 2e-16, which is
invisible to the converger.

The bit-identical energy across FULL_DOUBLE and FULL_DOUBLE+Kahan
confirms this — Kahan's compensation accumulators are accumulating
nothing meaningful, just running through the motions.

This generalises: Kahan is only useful when ε_machine is comparable to
or larger than the target tolerance. For FP64 + 1e-6 SCF threshold, it
is not.

## Why it costs so much wall time

The kernel grew 19 extra FP64 registers (1 scalar `c_w` + 1 dual `c_w2`
+ 6 vec3 compensators) per thread. The closed-shell GGA kernel was
already at 56 regs (56% occupancy on Pascal). Adding 19 regs almost
certainly forced register spills to local memory. Each `kahanAdd`
call also adds 4 FLOPs vs 1 for plain `+=`.

`gpu_compute_density` was 45% of GPU time before; with spills the kernel
became dominant enough to nearly triple wall time even though the rest
of the SCF (Coulomb, diag, DIIS) was unchanged.

## No bugs (b)

Both FULL_DOUBLE and FULL_DOUBLE+Kahan converge cleanly and
reproducibly. The hardened Kahan kernels produce the same final energy
as plain FULL_DOUBLE within 0.1 µHa run-to-run noise. No NaN, no
divergence, no failed convergence. Just no benefit.

## Conclusion

- Kahan in GPU kernels is dead in **both** precision modes:
  - Hybrid float32: breaks DIIS via FP-pattern shift (2026-03).
  - FULL_DOUBLE: zero accuracy gain, ~2.5× slower (this doc).
- `kahan.h` should remain in the tree for any future use case where
  ε_machine and tolerance are comparable, but **none of LIO's hot
  kernels qualify**.
- The note in MEMORY.md "Math-reordering opts now unlocked [under
  FULL_DOUBLE]" should be qualified: the unlocked space is for
  **structural reorderings** (FMA reorder, loop-tiling) that were
  unsafe in float32; it is **not** an unlock for compensated summation,
  which is also useless once you have FP64.

## What to try instead for register-bound kernels

`gpu_compute_density` (closed-shell GGA, 56 regs) and
`gpu_compute_density_opened` (open-shell, 93 regs) both have register
pressure that limits occupancy. The lever is **fewer accumulators**, not
more. See `research/gpu/optimize_open_shell_registers.md` (split open
shell into two closed-shell calls) and the in-progress closed-shell
register audit in `research/gpu/optimize_density_registers.md`.
