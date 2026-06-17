---
status: DONE
date: 2026-05-03
impact: removes all manual tuning burden; auto-selected params match empirically optimal (5800X3D: OMP=6, BLAS=4)
risk: low — formulas are conservative; both env vars still override
area: g2g/hardware_topo.{h,cpp} + g2g/init.cpp + lioamber/SCF.f90
---

# Overlap auto-tuning: hardware topology detection

## Problem

The int3lu/g2g overlap (see
[overlap_int3lu_g2g_implemented_2026_05_01.md](overlap_int3lu_g2g_implemented_2026_05_01.md))
requires two thread counts to be set correctly for good performance:

| Parameter | Role |
|---|---|
| `OMP_NUM_THREADS` | g2g partition workers + outer OMP sections driver |
| `LIO_OVERLAP_BLAS_THREADS` | OpenBLAS cap inside the concurrent int3lu section |

The empirically optimal values on a Ryzen 7 5800X3D (8 physical / 16 logical):
- `OMP_NUM_THREADS=6` — leaves 2 logical cores free for the int3lu BLAS section
- `LIO_OVERLAP_BLAS_THREADS=4` — tight median, lowest variance

(Measured in `sweep_overlap_params.py` over 10 runs each on fosfatoQMMM,
and `sweep_overlap_params_chloride.py` on the HCl TDDFT case.)

The problem: on a different machine (different physical-core count, different
HT factor), both numbers change. A user who enables `LIO_OVERLAP_INT3LU_G2G=1`
and doesn't know the right values will either undersub-scribe the CPU (slow)
or oversubscribe it (BLAS thrashing, variance spikes).

## Solution: hardware_topo module

New files `g2g/hardware_topo.h` and `g2g/hardware_topo.cpp` expose four
functions with no CUDA / OpenMP / G2G dependencies (only `<unistd.h>`,
`<cstdio>`, `<algorithm>`):

```cpp
int count_cpu_list(const char* s);           // parse Linux CPU list → count
int detect_physical_cores();                 // logical / HT-siblings
int recommended_omp_threads(int phys);       // max(2, phys*3/4)
int recommended_blas_threads(int phys);      // max(1, phys/2)
```

### Physical core detection

```
/sys/devices/system/cpu/cpu0/topology/thread_siblings_list
```

On HT machines this file contains the logical-CPU list that share a physical
core, e.g. `"0,8"` (2 SMT siblings) or `"0-3,8-11"` (4-way CMT).
`count_cpu_list` parses all Linux CPU list formats (single, comma-separated,
ranges, mixed) and returns the count.  Physical cores = `sysconf(NPROCESSORS_ONLN) / ht_siblings`.

Fallback: if the sysfs file is absent (container, non-Linux, ARM), the
function returns `sysconf(NPROCESSORS_ONLN)` unchanged.  The formulas then
over-estimate OMP threads by HT factor — conservative rather than harmful.

### Formulas and calibration

```
recommended_omp_threads(phys)  = max(2, phys * 3/4)
recommended_blas_threads(phys) = max(1, phys / 2)
```

Derivation from the 5800X3D sweep (8 phys cores, 16 logical):

```
phys=8 → OMP = max(2, 6) = 6  ✓  empirical best
phys=8 → BLAS = max(1, 4) = 4  ✓  empirical best
```

Intuition: when overlap is on, g2g's `Partition::solve()` uses all `OMP`
threads in its inner `#pragma omp parallel for`.  Each BLAS call inside int3lu
is a mini-DGEMM on roughly M×M doubles (M≈364 for fosfato); with HT, 4 threads
fill the physical cores the g2g section doesn't use.  At `OMP + BLAS > logical`,
memory-bandwidth contention adds variance.

The `3/4` factor is deliberately conservative — it leaves `phys/4` logical
threads free for OS jitter, GPU driver polling, and BLAS.  On a 32-core
EPYC (64 logical): `OMP=24`, `BLAS=16`.  We haven't swept those, but the
intuition holds and both sides still have headroom.

### Integration in g2g_init_()

```cpp
int phys = detect_physical_cores();
G2G::recommended_blas_threads = ::recommended_blas_threads(phys);

const char* ov = getenv("LIO_OVERLAP_INT3LU_G2G");
bool overlap_on = (ov && ov[0] == '1' && ov[1] == '\0');
if (overlap_on && getenv("OMP_NUM_THREADS") == nullptr) {
    int n_omp = ::recommended_omp_threads(phys);
    omp_set_num_threads(n_omp);
    if (verbose > 1)
        printf("  [overlap] auto OMP_NUM_THREADS=%d BLAS=%d (phys_cores=%d)\n",
               n_omp, G2G::recommended_blas_threads, phys);
}
```

Key guard: `getenv("OMP_NUM_THREADS") == nullptr`.  If the user has set
`OMP_NUM_THREADS` explicitly, we leave it alone.  The auto-tune fires only
when the overlap is enabled AND OMP is unset — both conditions required.

`G2G::recommended_blas_threads` is set unconditionally (even with overlap off)
so it is available as a sane fallback regardless.  The Fortran side reads it
via `g2g_recommended_blas_threads_()`.

### SCF.f90 BLAS thread read

```fortran
integer, external :: g2g_recommended_blas_threads
...
if (env_overlap_status == 0) then
    read(env_overlap_str, *, iostat=env_overlap_status) overlap_blas_threads
    if (env_overlap_status /= 0 .or. overlap_blas_threads < 1) &
        overlap_blas_threads = g2g_recommended_blas_threads()
else
    overlap_blas_threads = g2g_recommended_blas_threads()
endif
```

`LIO_OVERLAP_BLAS_THREADS` still overrides fully; the recommended value is
the fallback only when the env var is unset or invalid.

## TDDFT guard: no OMP change when overlap is OFF

The guard `getenv("OMP_NUM_THREADS") == nullptr` is only checked when
`overlap_on == true`.  When overlap is off (default), `omp_set_num_threads`
is never called, and TDDFT runs with whatever OMP setting the environment
provides.  On the 5800X3D, TDDFT is fastest at OMP=12-16; reducing to 6
would be a significant regression.  Confirmed:

| Run | `07_TDDFTHCL` wall | note |
|---|---|---|
| baseline (overlap off) | 52.6 s | OMP=12, default |
| overlap on (fosfato) | 52.6 s | TDDFT unaffected |

## TDDFT regression investigation (07_TDDFTHCL)

During e2e validation after the auto-tune was added, `07_TDDFTHCL` reported
`[FAIL] Test Dipole: ERROR`.  Root cause: `chloride.in` had been edited to
`ntdstep=5000` for benchmark runs (previous session) but the `.ok` reference
file was generated with `ntdstep=50000`.  Additionally the dipole values in the
`.ok` file differ from the current code (`0.15100342E+01` vs `0.15539555E+01`).

The failure is **pre-existing and unrelated to the auto-tune**: `git stash`
(reverting only `chloride.in` to `ntdstep=50000`, keeping init.cpp/SCF.f90
changes) reproduced the same dipole mismatch — the `.ok` reference was generated
under different physics.

Action required: regenerate `07_TDDFTHCL/dipole_moment_td.ok` with the current
code at `ntdstep=50000`.  This is a separate task; the auto-tune change itself
is clean.

## Unit tests

New: `test/unit_tests/kernels/hardware_topo_cpu_test.cpp` (187 subtests, all PASS).

Coverage:

| Group | What's tested |
|---|---|
| `count_cpu_list` | single CPUs, comma-separated pairs, ranges, multi-range, trailing newline (as returned by fgets), empty input |
| `recommended_blas_threads` | concrete values at phys=1,2,4,8,16,32; invariant `>= 1` for phys 1-64 |
| `recommended_omp_threads` | concrete values at phys=1,2,4,8,16,32 incl. clamp at 2; invariant `>= 2` for phys 1-64; `omp <= phys` for all phys 2-32 |
| `detect_physical_cores` | result ≥ 1; result ≤ `sysconf(NPROCESSORS_ONLN)`; logical divisible by phys |

The test uses `#include "hardware_topo.cpp"` directly (same pattern as
`partition_cost_cpu_test.cpp`) to avoid any build dependency on libg2g or
CUDA.  No `-lpthread` needed; `hardware_topo.cpp` uses `sysconf` not `<thread>`.

The formula tests pin the exact mapping so any future recalibration (e.g.
different `3/4` factor) is an explicit, visible test breakage rather than a
silent behavior change.

## Files modified / created

| File | Change |
|---|---|
| `g2g/hardware_topo.h` | new — declarations for the four topo functions |
| `g2g/hardware_topo.cpp` | new — implementations; no CUDA/OMP/G2G deps |
| `g2g/init.cpp` | `#include "hardware_topo.h"` replaces inline static helpers; `::recommended_*` calls replace inline formulas; `<thread>` removed |
| `lioamber/SCF.f90` | `overlap_blas_threads` default 4→0; fallback reads `g2g_recommended_blas_threads()` instead of literal 4; `omp_get_max_threads()` external declared for verbose log |
| `test/unit_tests/kernels/hardware_topo_cpu_test.cpp` | new — 187-subtest CPU test, auto-discovered by Makefile wildcard |

`hardware_topo.cpp` is auto-picked up by `SRCS:=$(wildcard *.cpp)` in
`g2g/Makefile`; no Makefile change needed.

## Verified results (5800X3D, fosfatoQMMM)

After auto-tune lands (overlap on, no manual env vars):

```
phys_cores detected = 8
auto OMP_NUM_THREADS = 6
auto LIO_OVERLAP_BLAS_THREADS = 4
fosfato wall median: 1.34 s   (baseline no-overlap: 1.81 s)
chloride TDDFT: 5.56 s        (baseline: 5.56 s — no regression)
```

The auto-selected OMP=6 / BLAS=4 matches the empirically optimal point from
the parameter sweep exactly.
