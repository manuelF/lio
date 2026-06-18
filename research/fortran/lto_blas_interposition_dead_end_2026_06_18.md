# LTO / -fexternal-blas / -fno-semantic-interposition — measured dead end (2026-06-18)

**Status: REJECTED for the default build.** Three "bigger-picture" gfortran
structural flags were enabled and measured end-to-end on the real cases. None
produced a measurable wall-clock win, and two perturb the FP-sensitive heme/
fosfato paths. This closes the structural-flag branch for lioamber.

## The root cause (the one finding to remember)

**No lioamber FFLAGS change can move fosfato (or the TD test cases), because
~95% of their wall is in separate compilation units:**

| Hot bucket (fosfato, per iter) | Time | Lives in | Built with lioamber FFLAGS? |
|--------------------------------|------|----------|------------------------------|
| Fock integrals (int3lu / Coulomb) | 309 ms (45%) | `libg2g.so` (CUDA) | **No** |
| Fock diagonalization (DSYEVD) | 181 ms (27%) | `libopenblas` | **No** |
| Base change / DIIS (DGEMM) | part of 141 ms accel-setup | `libopenblas` | **No** |
| Fortran glue (DIIS commut, update_emat, converger logic) | a few ms/iter | `liblio-g2g.so` | Yes |

Only the last row is lioamber-resident, and it is a thin few-ms/iter sliver.
The matmul-heavy code that `-fexternal-blas`/LTO *could* accelerate is the
TD/Ehrenfest path (`commutator.f90`, `calc_forceDS`, `ehrensubs`), which in the
test suite is small-M (chloride M~21) and host-bound by the Magnus propagator.

## Measurements (gfortran 13.3, RTX 3080 Ti + Ryzen 5800X3D, cuda=1 cpu=1)

Baselines: fosfato E = **-2341.457306** (24 iters); heme basin **-3139.2122**
(iter count is Lyapunov-chaotic run-to-run: observed 115 / 410 on the *same*
binary — energy basin is the only valid gate); chloride TD 50k steps = **15.4-
15.9 s**.

| Flag | fosfato E | heme basin | chloride TD | Verdict |
|------|-----------|------------|-------------|---------|
| `-fexternal-blas -fblas-matmul-limit=16` | -2341.457306 (bit-exact) | OK (132/160/168/444) | 15.9 s (noise) | No win; perturbs heme trajectory |
| `-flto=auto -ffat-lto-objects` (+ LFLAGS) | -2341.457305 (**1 µHa shift**) | OK (137/194) | 15.7/15.9 s (noise) | No win; breaks bit-exactness |
| `-fno-semantic-interposition` | not shipped | — | — | Same class as LTO; not separately tested |

`-flto` confirms the old NOTE in `Makefile.options`: cross-call inlining
exposes new FMA contraction and shifts fosfato off bit-exact, for zero wall
benefit. `-fexternal-blas` is bit-exact on fosfato (its matmuls reduce
identically) but matmul is absent from the per-iter SCF path, so it only
reaches TD/Ehrenfest where the suite is host-bound.

## When to revisit

- `-fexternal-blas` becomes worthwhile **only** if a large-M Ehrenfest/TD
  workload (M well above ~32, matrix-bound not host-bound) becomes the target.
  It would then convert `commutator.f90` etc. from gfortran's slow intrinsic
  matmul to OpenBLAS DGEMM (the 3-14x gap recorded in the DIIS-DGEMM work).
- LTO/interposition stay rejected unless the lioamber Fortran glue itself
  becomes a wall pole (it is not on any current case).

## What shipped instead

`Makefile.options`: added `-Wrealloc-lhs` alongside `-Warray-temporaries` under
`make lint=1`. These two **hidden-allocation** diagnostics are the only
remaining lioamber-side micro-perf lever with headroom (the SCF acceleration-
setup bucket, 141 ms/iter on fosfato). Current counts: array-temporaries 305,
realloc-lhs 374. They are staged for the same drive-to-zero-then-promote
treatment as the implicit-interface work (cee72a34), pending the file-by-file
cleanup. The NOTE in `Makefile.options` was rewritten with the measured
rejection rationale so the structural flags are not re-tried blindly.

## Health-flag pass (2026-06-18 follow-up)

Audited the remaining gfortran health/correctness flags. Outcome:

**Promoted to default FFLAGS as zero-count regression guards** (join
`-Wimplicit-interface`/`-Waliasing`):
- `-Wcompare-reals` — the lone fragile `REAL ==` was a geometry-fingerprint
  "unchanged?" test in `scf_extrapolation.f90`; replaced with an exact integer
  bit-compare (`transfer(fp,0_8) == transfer(geo_fp_last,0_8)`), provably
  identical for the finite non-±0 fingerprint. fosfato stays bit-exact.
- `-Wunused-dummy-argument` — the lone hit was a vestigial `ndiist` on
  `diis_fock_commut`; removed from the signature + 2 call sites. Behavior-
  preserving (arg was never referenced); fosfato bit-exact, heme basin OK.

**Two genuine uninitialized-use bugs fixed** (safety; both outside the e2e
suite so caught only by the flag):
- `RMMcalc3_FockMao.f90`: `Energy_Efield` was set only inside the `if(eefld_on)`
  branch but always added to `Energy` — garbage in the no-field Ehrenfest path.
  Now initialised to 0 before the branch (field-on path unchanged).
- `do_electronic_interpolation.f90`: `new_surf` was read after the substep loop
  but only assigned inside it — uninit if `tsh_Enstep<=0`. Now defaulted to
  `state_before` (no hop) before the loop.

**Left gated under `make lint=1`, documented as not cleanly promotable:**
- `-Wmaybe-uninitialized` (154): ~153 are false positives in the `faint_cpu`
  integral routines (`subm_int3mem` etc.) — hundreds of angular-momentum scalar
  temps the compiler cannot prove assigned-before-use per branch. Promoting
  would require mass-initialising those hot routines for no correctness gain.
- `-Wuse-without-only` (161), `-Warray-temporaries` (305), `-Wrealloc-lhs` (374):
  staged, file-by-file cleanup pending.

All curated CI e2e PASS (incl. open-shell 01_OxyMol); fosfato bit-exact
(-2341.457306); heme basin -3139.2122; 05/07 TD PASS.

## Correctness-flag sweep + runtime audit (2026-06-18, second follow-up)

Swept the bug-catching gfortran flags and ran two runtime audits. **No new bugs
found** beyond the two already fixed above — the Fortran layer is clean on every
path exercised.

**13 more correctness flags, all at zero, PROMOTED to default FFLAGS as guards**
(warning-only, no codegen/FP impact): `-Wline-truncation` (silent source drop),
`-Wcharacter-truncation`, `-Wsurprising`, `-Wreturn-type` (unset function
result), `-Wzerotrip`, `-Wundefined-do-loop`, `-Wdo-subscript` (compile-time
OOB), `-Wintrinsic-shadow`, `-Wconversion` (value-changing narrowing — note
`-Wconversion-extra` is 5075 and stays off as noise), `-Wreal-q-constant`,
`-Wtarget-lifetime`, `-Wc-binding-type`, `-Wampersand`.

**Two gated runtime-audit tiers added** (`make check=N`; zero default impact):
- `check=1`: `-fcheck=bounds,do,pointer,mem,recursion`.
- `check=2`: `-finit-real=snan -ffpe-trap=invalid` (uninitialised REAL used in
  arithmetic aborts with a backtrace).

Ran agua, fosfato, OxyMol (open), Fe3H2O6 (open TM), 04_2_ECP, TDDFTHCL, Heme
under **both** tiers: **all exit 0, zero bounds/pointer/memory violations, zero
FP traps, all energies correct.** This definitively confirms the 153
`-Wmaybe-uninitialized` hits in the `faint_cpu` integral routines are false
positives — heme/fosfato drive `int3mem`/`int3lu` hard with every uninitialised
real set to a trapping NaN and never trip. The audit tiers are kept in
`Makefile.options` for future use.

## `-Wall -Wextra -fimplicit-none` pass (2026-06-18, third follow-up)

`-Wall -Wextra` on the whole tree surfaced only **3 `-Wunused-variable`** (plus
the known 154 maybe-uninitialized): all dead `use, only:` imports —
`ndiis` in `diis_get_error`, `MM` in two `liomain` population/restart routines.
Verified each routine's body never references the symbol; removed them. Not
bugs, just dead imports. `-Wunused-variable` now zero and **promoted**.

`-fimplicit-none` (forbid implicit typing tree-wide) found exactly **one**
implicitly-typed symbol in the entire codebase: loop counter `i` in
`read_coords` (input_read.f90), whose routine lacked `implicit none`. Added
`implicit none` + `integer :: i`. Worked before via implicit-integer typing (not
a bug) but is exactly the silent-typo class the flag guards. `-fimplicit-none`
now compiles clean and is **promoted to default FFLAGS** — any future mistyped
name is a hard error, not a new silent variable. (Legacy files with an explicit
`implicit real*8(a-h,o-z)` rule are unaffected; the flag only bites truly
untyped symbols.)

Net default-build correctness guards now: `-Wimplicit-interface -Waliasing
-Wcompare-reals -Wunused-dummy-argument -Wunused-variable -fimplicit-none` plus
the 13 promoted earlier. fosfato bit-exact (-2341.457306); curated CI all PASS.
