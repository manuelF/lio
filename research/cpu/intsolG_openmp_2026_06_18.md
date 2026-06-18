# intsolG (QM/MM solvent gradients) OpenMP parallelization — DONE 2026-06-18

**Status:** DONE. Shipped. forces e2e PASS (1e-3 tol). Deterministic frc_mm + QM forces.

## TL;DR

`intsolG` (`lioamber/faint_cpu/subm_intsolG.f90`) computes the QM/MM solvent
force contribution. It was **fully serial** and is **the single largest cost in
the whole fosfato run** — 3.97 s (32% of total wall, 68% of the Forces section).
Parallelized over the MM-atom range with OpenMP:

| | before | after |
|---|---|---|
| QM/MM gradients | 3.97 s | **0.65 s** (−84%, ~6× on 8 cores) |
| Forces (total) | 5.85 s | 2.55 s (−56%) |
| **fosfato total** | **12.27 s** | **9.28 s (−24%)** |

forces/energy/mulliken/dipole e2e all PASS.

## Why it was slow and how it parallelizes

The routine is an Obara–Saika integral-gradient over basis-shell pairs; each
shell pair sums a contribution from **every MM point charge** (`ntatom-natom`,
here ~8800 for `nsol=2954`). The two inner `do iatom = natom+1, ntatom` loops per
shell pair (fill Boys-function scratch `s0s..x4x`, then accumulate forces) are the
hot spot and the Boys functions (`FUNCT`) are the bulk of the cost.

Each MM atom is **independent**, so we partition the MM-atom range across threads:

- **frc_mm rows** are written only by the owning thread → race-free and
  **bit-identical to the serial result** (no reduction on the large array).
- **frc_qm** is accumulated by every thread; each writes a private per-thread
  slot, summed back in **fixed thread order** after the parallel region
  (deterministic, `natom*3` is tiny).
- All integral **scratch** (`s0s..x4x`, ~90 temporaries) lives in a contained
  worker subroutine `intsolG_mm`, so each thread gets its own copies
  automatically as procedure locals — **no manual `private(...)` list to get
  wrong**, which is the usual source of OpenMP races. The shell-pair scalar
  setup (Q, Zij, term0, rmax cutoff) runs redundantly per thread but is cheap
  relative to the per-MM-atom Boys-function work it gates.

The read-only setup (`ns,np,nd,M2,Ll,SQ3`) and inputs are host-associated
(shared). The serial nuclear-repulsion preamble is left untouched.

## Numerics

- frc_mm: bit-exact vs serial.
- QM forces: deterministic run-to-run (fixed-order thread reduction). Residual
  run-to-run variation in the written `forces` file is **1e-6**, sourced upstream
  from the existing parallel-XC SCF density (energy already wobbles in its last
  printed digit) — not from intsolG. Baseline gap to `forces.ok` is 7.7e-5
  (float32 noise floor); forces test tolerance is 1e-3.

## Implementation

`lioamber/faint_cpu/subm_intsolG.f90`:
- `use omp_lib`; slim the host scope to setup + nuclear repulsion + the parallel
  driver; per-thread MM slice `[mm_lo, mm_hi]` via `omp_get_thread_num/num_threads`.
- Move the entire 6-section integral body (s|s … d|d) into a contained
  `subroutine intsolG_mm(mm_lo, mm_hi, frc_qm)`; the 12 `do iatom = natom+1,
  ntatom` loops become `do iatom = mm_lo, mm_hi`.
- Per-thread `frc_qm_threads(natom,3,nthr)` slots, fixed-order sum after the
  region.

No env toggle (ships unconditionally, per the no-toggle rule).

## Next levers in Forces (now 2.55 s)

- Coulomb gradients: 1.32 s — `int3G`, same shell-pair structure, likely the
  next OpenMP target.
- XC gradients: 0.39 s (already OMP via g2g).
- Nuclear attraction gradients: 0.16 s (`int1G`).
