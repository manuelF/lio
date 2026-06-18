# int3G (Coulomb gradients) OpenMP parallelization — DONE 2026-06-18

**Status:** DONE. Shipped. forces e2e PASS (1e-3 tol). Deterministic.

## TL;DR

`int3G` (`lioamber/faint_cpu/subm_int3G.f90`) is the three-center Coulomb force
gradient (density-fitting). It was **serial** and the 2nd-largest force cost after
[[intsolG_openmp_2026_06_18]]. Parallelized over the outer basis-shell loop of
each of its 18 shell-type blocks.

| | before | after |
|---|---|---|
| Coulomb gradients | 1.32 s | **0.19 s** (~7× on 8 cores) |
| Forces (total) | 2.55 s | 1.40 s |
| **fosfato total** | 9.28 s | **8.13 s** |

(Cumulative across the 2026-06-18 forces+overlap work: fosfato **12.27 → 8.13 s,
−34%**.) forces/energy/mulliken/dipole e2e PASS; open-shell Fe3H2O6 + agua PASS.

## Structure / approach

`int3G` = serial preamble (`int2G`, the XC-gradient `g2g_solve_groups`, and an
`igpu>2` GPU early-return) followed by **18 shell-block loops** `(ss|s)…(dd|d)`,
each an outer `do ifunct` × inner `do jfunct` × `do kfunct` over the auxiliary
density basis, accumulating into `frc(natom,3)`.

Same race-free recipe as intsolG, but parallelized over the **outer shell loop**
(not MM atoms, which don't appear here):

- The entire 18-block body moved into a contained worker `int3G_cou(frc)`; all
  ~600 scratch scalars **plus `Q`/`W`** (previously shared host allocatables,
  written every shell pair — would have raced) become procedure locals = auto
  thread-private.
- `!$omp do schedule(dynamic)` on each block's outer `do ifunct` loop distributes
  shell rows across the team (dynamic balances the triangular `jfunct=1,ifunct`
  loops).
- Per-thread `frc_threads(natom,3,nthr)` slots, summed in **fixed thread order**
  after the parallel region → deterministic (no `reduction` clause; each thread
  writes only its own slot during its `!$omp do` iterations).
- Read-only setup (`Jx, Ll, SQ3, ns..ndd`) host-associated; the serial preamble
  and the `igpu>2` early return are unchanged (now also `deallocate(Jx,Ll)` on
  that early-return path).

## Numerics

Run-to-run forces variation 1e-6 (upstream parallel-XC SCF density noise, same as
the serial build); baseline gap to `forces.ok` 7.7e-5; tolerance 1e-3.

## Next Forces levers (Forces now 1.40 s)

- XC gradients 0.39 s — already OMP (g2g).
- `int1G` nuclear-attraction gradients 0.16 s — small; same shell-pair shape if
  ever needed.
- Forces is now 17% of total (was 48%); the SCF XC solve is again the pole.
