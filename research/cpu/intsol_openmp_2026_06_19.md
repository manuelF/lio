# intsol (QM/MM 1-e Fock) OpenMP — DONE 2026-06-19

**Status:** DONE. Bit-exact Hmat, 5.7× on the routine, fosfato total wall −11%.

## Context / profile

After the recurring XC-Fock pole was shown to be near-floor (BLAS3 dead — see
[[density_blas3_dead_end_2026_06_19]] — and the CPU group solve is load-balanced
to imbalance≈1.08), the next-biggest fosfato cost is a **one-time** item in
*Initialize SCF*:

```
1-e Fock                 1.212 s  (58.5% of Initialize SCF)
  Nuclear attraction     0.043 s  (int1, 34 QM nuclei)
  QM/MM (intsol)         1.169 s  (96.4% of 1-e Fock)   <-- fully serial
```

`intsol` builds the QM/MM core-Hamiltonian (point-charge attraction) over
`nsol=2954` MM charges. It had **zero OpenMP** — the exact situation of the
already-parallelized `intsolG` ([[intsolG_openmp_2026_06_18]]) and `int3G`
([[int3G_openmp_2026_06_18]]).

## Why the parallelization axis differs from intsolG

`intsolG` partitions the **MM-atom** range because its output `frc_mm(jatom,:)`
is *per MM atom* (disjoint rows) — bit-exact with no reduction.

`intsol`'s output `Hmat(vecmat_ind)` is a **sum over MM atoms** into packed
basis-pair indices: `tna = sum_iatom(...); Hmat += ccoef*tna`. Partitioning MM
atoms would split that sum into per-thread partials and distribute the `ccoef`
multiply → **reordered FP, not bit-exact**. Wrong axis here.

The bit-exact axis is the **outer basis-shell loop `ifunct`** (the `int3G`
axis). `vecmat_ind` encodes the `(ifunct,jfunct)` packed position with
`jfunct<=ifunct` only, so distinct `ifunct` → **disjoint Hmat indices**, and
every index is still accumulated in the same `nci/ncj/l1..l4` order as serial.
Hmat is therefore **bit-exact**; the MM sum stays serial inside each thread.

## Implementation

One `!$omp parallel default(shared) reduction(+:E1s) private(scratch + all
scalar temps)` region wraps all 6 shell-blocks (ss,ps,pp,ds,dp,dd); each block's
outer `do ifunct` gets `!$omp do schedule(dynamic)`. Scratch `s0s..s4s(ntatom)`
moved inside the region (allocated per thread → auto-private). `E1s` is a scalar
energy reduction — reordered (~1e-12 rel), but its e2e tolerance is 1.5e-2 Ha and
it does **not** feed Fmat, so SCF trajectory is untouched. ~35 LOC, all in
`lioamber/faint_cpu/subm_intsol.f90`.

## Results (fosfato, RTX-less CPU-only build, Ryzen 5800X3D)

| metric | before | after | |
|---|---|---|---|
| QM/MM intsol | 1.169 s | 0.205 s | **5.7×** |
| 1-e Fock | 1.212 s | 0.248 s | −80% |
| Initialize SCF | 2.081 s | 1.100 s | −47% |
| SCF total | 5.78 s | 4.93 s | −15% |
| **total wall** | **~7.30 s** | **~6.48 s** | **−11%** |

- SCF **27 iterations preserved** across OMP={1,4,8} (Hmat bit-exact → trajectory
  invariant); energy varies only in the last digit (pre-existing float32 XC noise).
- e2e: fosfato (energy/forces/mulliken/dipole) + full CPU travis subset incl.
  `06_QMinPcharges` (another intsol path) all PASS.

## What's NOT a lever (measured this session)

- **BLAS3 density matvec**: re-tested at fosfato's large group shapes
  (m up to 245, np up to ~10k — the top-20 groups carry 80% of cost, a totally
  different regime from the m=21 TD case). Full-path microbench (matvec +
  W-materialization + a realistic epilogue) = **0.4–0.6×, slower**. The SGEMM is
  fast (17 ms at m=245) but materializing W (10·np·m) and streaming it through the
  epilogue (104 ms) kills it — the hand kernel's register-fused matvec+epilogue
  (zero W traffic) wins at every m. Confirms [[density_blas3_dead_end_2026_06_19]]
  generalizes to large groups. **Method: /tmp/densbench.cpp times the FULL path,
  not just the matmul (the original bench's flaw).**
- **CPU group load balance**: steady-state imbalance = max·nbins/Σ ≈ **1.08**
  (rebalancer handles the giant np=7507/9902 groups well); ≤8% headroom, not worth
  the bit-exactness risk of splitting groups.

## Next levers

The recurring per-iter **XC-Fock (g2g density, 2.87 s, 40% of CPU)** is the pole
again and is near-floor for this structure. The only remaining big levers are
SCF iteration-count reduction (heme-style, see convergence/) or a fundamentally
different density-eval algorithm (none found).
