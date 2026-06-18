# lioamber Makefile optimizations + interface-lint program (2026-06-16)

Status: **implicit-interface 809 → 0 (-100%)** and **-Waliasing 0**, all e2e
PASS (agua, Fe3H2O6, fosfato, TDDFTHCL, 04_ECP, 04_2_ECP). Both flags
**promoted to the default FFLAGS** (always-on regression guard); only
`-Warray-temporaries` (305) remains `lint=1`-gated. The final 86 were closed
by adding explicit-interface blocks for the LIO-internal bare externals rather
than modularizing the ECP/driver files: the self-reference problem is avoided
with the rule "a file that both defines and calls a procedure imports its
*callees* via `use lio_interface, only: ...` so its own name never enters its
definition scope" (declaration-only, no link-symbol change). `scf` needs
`type(operator)`, so it lives in a separate `scf_interface` module to avoid a
module cycle through the low-level `liosubs_math`/`packed_storage` users of
`lio_interface`. Scope: `lioamber/Makefile*`, compiler-flag audit, and the
`lint=1` warning workflow ("add as lints → fix → promote to main flags") — now
fully exercised end to end for the implicit-interface class.

## Interface modules added (foreign-function + bare-external boundary)

| Module | Covers | Warnings |
|--------|--------|----------|
| `gpu_timers_interface` | 7 `g2g_timer_*` C hooks | 336 |
| `linalg_interface` | 22 BLAS/LAPACK routines | ~110 |
| `gpu_interface` | 49 `g2g_*`/`aint_*`/`int3lu_gpu_*` C hooks | ~145 |
| `openblas_interface` | `openblas_set/get_num_threads` | 7 |
| `omp_lib` (stdlib) | `omp_*` | ~28 |
| `packed_storage_interface` | `spunpack/sprepack` family | 77 |
| `lio_interface` | `liocmplx` (+ future bare externals) | 36 |

**FP/ABI findings:** (1) explicit interfaces are declaration-only but can shift
FMA contraction in *caller* code via `-fipa-pta`; heme iter count jitters within
its documented chaos band (138–142) across rebuilds — basin always correct, e2e
always pass; fosfato final-energy ulp jitter is the pre-existing partition-timing
nondeterminism, not the interfaces. (2) `-fallow-argument-mismatch` does NOT mask
explicit-interface TKR errors, so the compiler reliably caught every wrong
rank/type — used as the oracle. (3) Several C `double*` args are scalar sentinels
(`g2g_solve_groups(..,0)`, `aint_qmmm_forces(ff1G,0)`); call sites changed to
type-correct `0.0D0` / array-element placeholders, behavior preserved (fosfato
forces e2e identical). (4) `g2g_exact_exchange_open` callers pass a 5th arg the C
side ignores — declared to match callers.

## Remaining 86 (deferred — structurally distinct)

All LIO-internal bare subroutines, dominated by the ECP cluster in
`generalECP.f90` / `readECP.f90` (`write_post`×13, `write_*`, `obtain*`,
`norm_c`, `read_ecp`, …) plus driver subs (`scf`×12, `liomain`,
`init_lio_common`, `dft_get_*`, …). These files **both define and call** the
same procedures, so a single interface module hits a self-reference error; the
correct fix is to **modularize** those files (module procedures get auto-correct
interfaces + intra-module visibility) — a higher-risk refactor on FP-fragile,
less-covered ECP paths, to do with `04_*_ECP` in the validation loop. The `scf`
interface additionally needs `type(operator)` (import) + optional args.

Scope: `lioamber/Makefile*`, compiler-flag audit, and the `lint=1` warning
elimination workflow requested ("add as lints → fix → promote to main flags").

## Profiling context — where fosfato actually spends time

`LIO_OVERLAP_INT3LU_G2G=1`, fosfato single point, 24 iters, ~24 ms/iter:

| Phase | ms/iter | Note |
|-------|---------|------|
| fock  | 10.3 | `int3lu`=1.0 (CPU, GPU-offloaded) ‖ `g2g` XC=10.2 (GPU) → **CPU idle 9.2** |
| diag  | 7.1  | DSYEVD, OpenBLAS floor (see `scf_diag_lapack_floor`) |
| accel | 4.5  | DIIS commutator + base change (already BLAS-3) |
| build/rest | ~1.8 | scalar glue |

**The structural pole is the 9.2 ms/iter CPU idle** waiting on the GPU XC
kernel. Everything downstream (diag, DIIS) needs the *complete* Fock matrix,
which is not ready until `g2g` finishes — so the idle window cannot be filled
with same-iteration CPU work. This is algorithmic/scheduling, **not** a
Makefile change. Compiler flags touch only the ~1.8 ms of scalar glue.

## The 10+ Makefile-derived optimizations (audit)

Mined from `lioamber/Makefile.options`. Status as of this note:

**SHIPPED (already in default FFLAGS, verified bit-exact earlier):**
1. `-O3` global + selective `-O1` for FP-sensitive integral files (private_flag).
2. `-march=native -mtune=native` (overridable via `ARCHFLAGS_FC` for release).
3. `-fipa-pta` (interprocedural points-to → better alias analysis).
4. `-floop-nest-optimize -fgraphite-identity` (polyhedral loop nest opt).
5. `-fprefetch-loop-arrays` + `-funroll-loops`.
6. `-fstack-arrays` (heap→stack for automatic arrays; guarded re: large temps).
7. `-fno-math-errno -fno-trapping-math` (drop errno/trap glue; no reassoc).
8. `-fno-plt -falign-functions=32 -falign-loops=32` (call/branch overhead).
9. `-ffrontend-optimize` (front-end array-op simplification).
10. Link hardening / load speed: `-Wl,--as-needed -Wl,-O1 -Wl,-z,relro
    -Wl,-z,now -Wl,--sort-common`.

**REJECTED (FP-perturbation hazard on the Lyapunov-sensitive SCF; documented):**
- `-flto=auto`, `-fno-semantic-interposition`: cross-module inlining exposes new
  FMA contraction → bit shifts; re-validate heme OMP={1,4,6,8} before adding.
- `-ffast-math`/`-Ofast`/`-fexternal-blas`: reassociate FP, forbidden.

**WARNING-ONLY diagnostics (gated behind `lint=1`, the subject of this note):**
11. `-Warray-temporaries` — surfaces hidden heap alloc+copy at call sites
    (real perf signal; 305 hits).
12. `-Wimplicit-interface` — surfaces unchecked Fortran/C and Fortran/BLAS
    boundaries (809 hits; ABI-safety signal).
13. `-Waliasing` — surfaces argument aliasing (88 hits).

The genuine *perf* lever among the warning flags is `-Warray-temporaries`;
the others are correctness/ABI hygiene. None changes codegen, so they stay
`lint=1`-gated per the standing toggle pattern until a class reaches zero, at
which point that class's flag can graduate to the default build.

## Interface-lint workflow — Phase 1a result

Created `lioamber/gpu_timers_interface.f90`: a module of explicit interfaces for
the seven `g2g_timer_*` C hooks (`extern "C"` in `g2g/timer.cpp`; they take
`(const char*, unsigned length)`, and gfortran auto-passes the hidden length for
a `character(*)` dummy, so the interface matches the existing calls exactly —
declaration-only, zero codegen/FP impact).

Wired `use gpu_timers_interface` into **all 42 timer-calling files** (module
scope for module files; before each subroutine's `implicit none` for standalone
files; aggregator objects `ehrensubs/excitedsubs/converger_subs/properties` for
`#include`d subdir files). Dependency block added in `Makefile.depends`.

**Result:** g2g_timer implicit-interface warnings **336 → 0**. Total
implicit-interface 809 → 471. **Bit-exact** (fosfato 24 iters, E=-2148.6510065).
e2e PASS: agua (closed), Fe3H2O6 (open-shell restart), TDDFTHCL (TD).

## Remaining warning inventory (sized for Phase 1b / 2)

Implicit-interface 471 left, dominated by two clean foreign-boundary classes:
- **BLAS/LAPACK (~110)**: dgemm 68, dsyevd 8, cgemm 5, zgemm 4, dgels 4,
  dpotrf 3, + ~20 more routines. Fix = one `blas_interface`/`lapack_interface`
  module. **Risk**: each signature must match exactly; a latent type mismatch in
  existing code becomes a hard compile error (could surface a real bug or break
  the build). Wire ~30 files. Defer unless validated incrementally.
- **Internal LIO externals (~360)**: liocmplx 36, spunpack/sprepack ~74, write_*,
  aint_* ~40, int3lu_gpu_* ~10, scf/liomain/drive, etc. Fix = `use` the defining
  module or add interfaces. Spread across ~50 scoping units.

**Promotion gate:** `-Wimplicit-interface` moves to default FFLAGS once ALL
classes reach zero. **DONE** — timers ✓, BLAS ✓, internal ✓ (incl. the ECP /
driver self-reference cluster). `-Wimplicit-interface` and `-Waliasing` now
ship in the default FFLAGS (`lioamber/Makefile.options`) as always-on guards;
`-Warray-temporaries` is the only flag left under `lint=1`.

- **Phase 2 (array-temporaries, 305)**: the perf-relevant class. FP-fragile —
  rewriting call sites can reorder reductions and double heme iters. Gate every
  change on heme OMP={1,4,6,8} median + fosfato bit-exact. Hot-path candidates
  for fosfato: subm_int3lu (3), commutator (7), basechange_gemm (1),
  matrix_diagon_dsyevd (8). Most of the 305 are in TD/ehren/excited/cdft (cold
  for fosfato).

## Files

- `lioamber/gpu_timers_interface.f90` (new module)
- `lioamber/Makefile.depends` (OBJECTS + dependency block)
- 42 timer-calling files: `use gpu_timers_interface` added.
