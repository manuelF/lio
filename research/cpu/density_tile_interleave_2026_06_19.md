# Density kernel: interleave the transposed tile to kill pointer spills — DONE 2026-06-19

**Status:** DONE. Shipped. Bit-exact (agua/Fe3H2O6/fosfato e2e PASS).

## Disassembly finding

Disassembling `cpu_compute_density_gga_batch<float>` (the density matvec, 59% of
the TD-on-CPU `solve`) showed the hot inner loop was **load-port bound**: every
`vfmadd231ps` was preceded by a `mov rax,[rsp+…]` — **20 stack pointer-reloads per
20 FMAs** (1:1).

Root cause: the 10 GGA channels (value + 3 grad + 6 hessian) lived in 10 separate
`[m*B]` sub-arrays at `Tk[k] = base + k*(m*B)`. Because `m` is a **runtime**
argument, the channel stride `k*(m*B)` is not a compile-time displacement, so the
compiler materialized 10 runtime base pointers — which don't fit in GP registers
and spilled to the stack, reloaded every iteration. Each FMA's address depended on
the just-loaded pointer (extra latency too).

## Fix

Interleave the tile as `T[(j*NCH + k)*B + b]` (NCH=10, B=8 both constexpr). The
per-channel offset becomes a **compile-time** `k*B`, so the 10 FMAs address
`[Tj + k*B]` off ONE base register:

```
old:  mov rax,[rsp+0x128]; vfmadd231ps ymm10,ymm1,[rax+rdx]   (x10, spills)
new:  vfmadd231ps ymm2,ymm1,[rdx]; vfmadd231ps ...,[rdx+0x20]; …  (one base)
```

Inner-loop pointer-reloads: **20 → 0**. The loop is now FMA-bound. Same values,
same summation order ⇒ **bit-exact** with the per-point reference kernel.

## Result

Alternating .so-swap A/B (cancels machine-load noise), chloride.in 3000 steps,
steady state: **TD-Pred-XC 7.27 → 6.75 s (~7%)**. Less than the naive load-port
estimate (~1.3×) — Zen3's 3 load ports partly absorbed the extra loads, and the
interleaved transpose has slightly worse write locality (stride NCH·B vs B), which
offsets a little. Still a free, bit-exact win on the shared CPU XC kernel (SCF + TD).

## Remaining (not pursued)

- The 10 `acc_*[B]` output accumulators (kept across the i-loop) + the 10 W vectors
  exceed 16 ymm, so the accumulators still spill to stack between i-iterations
  (~92 `vmovaps`). Inherent to the 10-output structure; not cleanly removable.
- The PBE functional (`calc_ggaOS`, the other 41% of solve) is scalar
  transcendentals — only vector-math/approximation would help (out of scope).
