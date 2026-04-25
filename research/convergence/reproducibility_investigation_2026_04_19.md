# Reproducibility and FULL_DOUBLE Bisection — 2026-04-19

**Goal**: Answer two questions precisely, with measurements:
1. What are the distinct sources of run-to-run non-reproducibility on the
   float32 GPU path, and how much does each contribute?
2. What is the actual scope of the `full_double=1` breakage (first reported
   in `numerical_stability_investigation_2026_04_17.md`) — is it a universal
   regression, or scoped to something specific?

**Status**: Both questions answered. FULL_DOUBLE scope narrowed to a
system-size threshold (works for agua/3 atoms, broken for fosfato/34 atoms).
Root cause still unknown; ruled out memory safety, QM/MM-only, and build-flag
mix-ups.

---

## Part 1 — Float32 GPU reproducibility

All runs: fosfatoQMMM, 34 QM atoms, `liosolo -i <in> -b basis -c fos.xyz`,
RTX 3080 Ti, HEAD of `optimizations` branch, clean `make cuda=1 cpu=1` build.

### Experimental matrix (5 configurations × 3–5 runs each)

| Configuration | Runs | Iters | Energy spread (Ha) | Mean energy (Ha) |
|---|---|---|---|---|
| `OMP=15`, `fgm=0.3` (default cache) | 5 | 25 (×4), 24 (×1) | **1.1e-6** (4 runs) + **40e-6 outlier** | −2148.6509x |
| `OMP=15`, `fgm=0.0` (no cache) | 3 | 25 (×3) | **2e-7** | −2148.65096 |
| `OMP=1`, `fgm=0.3` | 3 | 25 (×3) | **3e-7** | −2148.65097 |
| `OMP=1`, `fgm=0.0` | 3 | 25 (×3) | **3e-7** | −2148.65097 |

### Hypothesis ledger

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | GPU `atomicAdd(double)` in `gpu_scatter_rmm` floors non-determinism | **Confirmed** | 3e-7 residual spread with OMP=1 + fgm=0 — the only remaining source in that config is GPU warp scheduling inside the scatter kernel |
| H2 | Partition `rebalance()` amplifies noise via timing-dependent group→thread reassignment | **Confirmed** | fgm=0.3 → 4× wider spread than fgm=0.0 (same OMP). Matches prior investigation |
| H3 | OpenMP `reduction(+:energy)` at `partition.cpp:557` adds noise | **Confirmed** | OMP=1 tightens spread ~3× AND shifts the mean by ~3e-6 Ha (different partition of 303 CPU groups over threads → different FP summation order) |
| H4 | Cached `fgm` causes occasional 24-iter early convergence | **Confirmed** | 1 of 5 runs in the default config hit 24 iters at −2148.65092 (~40e-6 Ha off), none of the non-cached runs did. Rebalance tips the system across the convergence threshold one iter early |

### Minimum achievable spread: 3e-7 Ha

With `OMP_NUM_THREADS=1` and `free_global_memory=0.0`, the residual is 3e-7 Ha
over 3 runs. This is the irreducible floor attributable to GPU scheduling
inside `gpu_scatter_rmm`'s `atomicAdd(double)` — the only non-deterministic
operation remaining when CPU threading and cache-driven rebalance are both
disabled.

Going below 3e-7 would require either:
- **Deterministic GPU scatter** (sort indices + serial prefix-sum; costs perf)
- **Moving to FULL_DOUBLE** (would drop the float32 noise floor entirely —
  blocked on the bug below)

---

## Part 2 — Ground-truth energy comparison

### Energy ladder on fosfato (converged SCF)

| Build | Mean energy (Ha) | Iters | Δ from oracle (Ha) |
|---|---|---|---|
| **CPU FULL_DOUBLE (oracle)** | **−2148.6509176** | 27 | — |
| CPU float32 | −2148.6508841 | 27 | +3.4e-5 |
| GPU float32 (OMP=15, fgm=0.3) | −2148.6509632 | 25 | **−4.6e-5** |
| GPU float32 (OMP=1, fgm=0.0) | −2148.6509664 | 25 | **−4.9e-5** |
| GPU FULL_DOUBLE | −2148.6443x (unstable) | 30–37 | **+6.6e-3** (broken) |

Two observations:

1. **Float32 GPU is closer to the double-precision oracle than float32 CPU is**
   (−4.6e-5 vs +3.4e-5). Both are within ~5e-5 Ha — well inside chemical
   accuracy and consistent with grid-integration discretization error being
   larger than float32 roundoff.

2. **FULL_DOUBLE GPU on fosfato is off by 6.6 mHa** — 100× the float32 error
   and 200× the CPU float32 error. This is the bug.

---

## Part 3 — FULL_DOUBLE scope bisection

### Agua (`00_agua`, 3 atoms, neutral closed-shell, no MM)

- **CPU FULL_DOUBLE**: converged, 14 iters, Total = −76.066854 (matches `output.ok` exactly)
- **GPU FULL_DOUBLE**: 3/3 bit-exact runs, Total = **−76.066854**, matches CPU FD
- **GPU float32**: 3/3 bit-exact runs, Total = −76.066857 (3e-6 from oracle)

**Agua FULL_DOUBLE works correctly and is bit-exactly deterministic.**

### Fosfato variants (`03_fosfatoQMMM`, 34 QM atoms, charge −1)

| Variant | Iters | Energy range (Ha) | Status |
|---|---|---|---|
| with MM (nsol=2954), `precision=1` | 30–37 | 120e-6 spread, wrong by 6.6 mHa | Converges but wrong |
| without MM (nsol=0), `precision=1` | 100+ (never) | 186e-3 spread | Doesn't converge at all |
| without MM (nsol=0), float32 | 25 | — | Baseline works |

**Removing MM makes FULL_DOUBLE worse, not better.**

### Per-step divergence (fosfato, GPU FD vs CPU FD)

| Step | CPU FD | GPU FD | GPU float32 |
|---|---|---|---|
| 1 | −1440.65477 | −1440.65173 | −1440.65145 |
| 2 | 403.86419 | 403.96535 | 404.01040 |
| 5 | −1566.53810 | −1564.76404 | −1566.36380 |
| 6 | −1817.30115 | −1820.96865 | −1817.43528 |
| 7 | −1802.56346 | **−1768.08232** | −1802.26246 |

**GPU FULL_DOUBLE diverges catastrophically at step 6–7** (off by 3–34 Ha)
while both CPU FD and GPU float32 track each other within 0.3 Ha. This is
where a bad Fock/DIIS extrapolation sends the trajectory off, and it doesn't
recover.

### Hypothesis ledger for the brokenness

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| B1 | Wrong build flag (`full_double=1` vs `precision=1`) | **Refuted** | Both give identical broken behavior; `precision=1` just sets `full_double=1` + toggles unrelated TD_SIMPLE in lioamber |
| B2 | Texture channel-format mismatch in `iteration.cu:272`,`:760` | **Unlikely** | The same code path runs agua correctly — int2 packing of doubles works end-to-end there |
| B3 | AINT QM/MM template instantiation ABI issue | **Refuted** | Fosfato with nsol=0 (no QM/MM) is *more* broken than with nsol=2954 |
| B4 | Memory-safety bug (OOB, uninit) | **Refuted** | `compute-sanitizer --tool=initcheck` **and** `--tool=memcheck` both return 0 errors on fosfato GPU FD |
| B5 | libxcproxy stubs fire in non-libxc build | **Refuted** | Built without libxc; the `engañar al compilador` stubs are only compiled when `USE_LIBXC=1` |
| B6 | Scales with system size / basis count | **Supported** | Agua (3 atoms, 19 bf) correct; fosfato (34 atoms, 357+ bf) broken |
| B7 | Specific to closed-shell anions or specific elements (P, O) | **Not tested** | Agua is closed-shell neutral; need intermediate tests |

### Next bisection candidate

`02_Fe3H2O6` (19 atoms, open-shell, charge +3) sits between agua and fosfato
in size. Not run in this session because it's open-shell (different GPU
code path via `solve_opened`), which confounds the size-vs-code bisection.

Better intermediate candidates:
- Shrink fosfato to 5–10 atoms (just PO4 anion + a couple waters QM) and see
  at what size GPU FD starts diverging.
- `01_OxyMol` (O2-type closed-shell molecule): check whether it reproduces
  agua's correctness.

---

## Part 4 — FULL_DOUBLE breakage is NOT pure size-dependence (extended bisection)

Added 2026-04-19 (same day): ran a broader matrix of small closed-shell molecules
with `int_basis=t`, `told=1e-7`, `ndiis=30`, `gold=10`, `nsol=0`, `OMP=1`, `fgm=0`
(i.e. the cleanest setup — no MM, no caching, deterministic threading). FD vs
float32 comparison across 10 systems.

| System | Atoms | Basis fns | FD iters (3 runs) | FD spread (Ha) | Δ(FD−f32) Total (Ha) | Δ(FD−f32) XC (Ha) | Verdict |
|---|---|---|---|---|---|---|---|
| NH₃ | 4 | 21 (1d) | 11, 11, 11 | 0 | — | — | **BIT-EXACT** |
| CH₄ | 5 | 23 (1d) | 9, 9, 9 | 0 | — | — | **BIT-EXACT** |
| H₂O₂ | 4 | 34 (2d) | 13, 13, 13 | 0 | — | — | **BIT-EXACT** |
| H₂CO | 4 | 34 (2d) | 15, 15, 15 | 0 | — | — | **BIT-EXACT** |
| CH₃F | 5 | 36 (2d) | 17, 17, 17 | 0 | — | — | **BIT-EXACT** |
| CH₃OH | 6 | 38 (2d) | 15, 15, 15 | 1e-6 | — | — | **BIT-EXACT** |
| HCOOH (planar) | 5 | 49 (3d) | 200*, 153, 75 | **1.0 mHa** | **+127.6 mHa** | **+127.5 mHa** | **CATASTROPHIC** |
| HCOOH (non-planar) | 5 | 49 (3d) | 181, 29, 170 | 1.2 mHa | +127.6 mHa | (same) | **CATASTROPHIC** (planarity not the cause) |
| CH₃COOH | 8 | 68 (4d) | 34, 38, 42 | 0.4 mHa | +0.8 mHa | +0.1 mHa | **JITTERY** (small offset, big iter spread) |
| H₃PO₄ | 8 | 85 (5d) | 16, 17, 17 | 5e-6 | −0.3 mHa | −0.3 mHa | **BIT-EXACT** (mild offset from f32) |

(*: HCOOH run 1 hit nmax=200 without "Convergence achieved"; others converged.)

### Findings

1. **Not pure size-dependence.** H₃PO₄ (85 bf, 5 d-functions) works bit-exactly;
   HCOOH (49 bf, 3 d-functions) is catastrophically broken. Basis-function count
   is not the trigger.

2. **HCOOH is the smallest clear failure.** 5 atoms, all three runs converge
   (or don't) to the same wrong state, off by 128 mHa from float32. Run-to-run
   spread within FD is 1 mHa — FD is self-inconsistent but systematically
   wrong-inconsistent.

3. **The error concentrates in the XC term for HCOOH.** Contributions (in Ha):
   | Term | float32 | FD | Δ |
   |---|---|---|---|
   | Total | −189.573 | −189.446 | **+127.6 mHa** |
   | One-electron | −398.473 | −398.453 | +19.4 mHa |
   | Coulomb | +161.597 | +161.577 | −19.4 mHa |
   | Nuclear | +70.301 | +70.301 | 0 |
   | XC | **−22.998** | **−22.871** | **+127.5 mHa** |

   1-e and Coulomb shifts cancel exactly (same density difference, different
   operator). The XC term carries essentially all the error.

4. **Fosfato's 6.6 mHa error is *milder* than HCOOH's 127 mHa error,** even
   though fosfato is 7× bigger. This rules out "scaling with size" as a
   simple model.

5. **Acetic acid (contains the COOH group) is mildly broken** (0.8 mHa, big
   iter jitter), HCOOH (also COOH) is catastrophic, H₃PO₄ (different O arrangement)
   is fine. COOH-like O arrangement is a strong predictor but not the whole story
   (Fe₃H₂O₆ has no COOH and is also broken; fosfato contains phosphates *and*
   an acetate).

### Updated hypothesis ledger for FULL_DOUBLE brokenness

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| B1 | Wrong build flag (`full_double=1` vs `precision=1`) | **Refuted** (earlier) | Both behave identically |
| B2 | Texture channel-format mismatch | **Unlikely** (earlier) | Agua fine end-to-end |
| B3 | AINT QM/MM template instantiation | **Refuted** (earlier) | Fosfato without MM is *more* broken |
| B4 | Memory safety (OOB, uninit) | **Refuted** (earlier) | memcheck + initcheck clean on fosfato |
| B5 | libxcproxy stubs | **Refuted** (earlier) | Built without libxc |
| B6 | Scales with system size / basis count | **Refuted** (this part) | H₃PO₄ (85 bf) works; HCOOH (49 bf) catastrophically broken |
| B7 | Specific to anions or specific elements (P, O) | **Partial** | H₃PO₄ fine; HCOOH broken. Element identity insufficient |
| **B8** | **Triggered by specific XC grid / density patterns in certain molecules** | **Most consistent with data** | Error concentrates in XC term for HCOOH; broken set shares O-rich planar-ish regions with specific density gradients |
| B9 | DIIS path-dependence exposes FD-amplified noise | **Possible** | HCOOH iters vary 75–200; acetic 34–42. FD makes the SCF *less* stable, not more, in these cases |

### What to try next

- Run `gpu_compute_density` in isolation on HCOOH's converged float32 density
  with both FD and f32 builds; compare the computed XC energy directly without
  SCF feedback. Locates whether the bug is in `compute_density` (the GGA kernel)
  or in `accumulate_point` (the exchange-correlation integrator).
- Check if disabling the grid's fine shell or reducing `max_function_exponent`
  makes HCOOH converge correctly with FD — would pin it to far-field integration.
- Git-bisect FD correctness against `01_OxyMol` and `00_agua` over the last ~2
  years of commits to find when FD last worked at scale (the problem may be a
  latent regression, not an architectural limit).

---

## Part 5 — Grid-dependence of the FD XC error (smoking gun)

Added 2026-04-19: HCOOH FD XC energy vs grid density.

| Grid config (cube, sphere, exp) | float32 XC (Ha) | FD XC (Ha) | Δ(FD−f32) |
|---|---|---|---|
| Default (5.6, 0.6, 10) | −22.998467 | **−22.870929** | **+127.5 mHa** |
| Dense  (3.0, 0.2, 15)   | −22.998452 | **−22.974199** | **+24.3 mHa** |
| Coarse (12.0, 2.0, 6)   | −22.998431 | **−22.737938** | **+260.5 mHa** |

**float32 XC is grid-invariant** (max spread 36 μHa across all three configs —
consistent with normal discretization error). **FD XC varies by 260 mHa with grid**
— 6-order-of-magnitude anomaly.

### Interpretation

A *correctly* implemented XC integrator should give the *same* XC energy
(to discretization accuracy) regardless of grid density. Float32 does this.
FD does not on HCOOH.

This rules out "FD gives the true answer and f32 is wrong from roundoff" —
the opposite is true. The FULL_DOUBLE XC kernel path is computing systematically
wrong values on HCOOH, and the error magnitude depends on the grid partition.
The error is *not* random noise (3 runs at each grid agree to ~1 mHa with each
other) — it's a deterministic bug that produces a different (wrong) answer per
grid.

### Implications for the bug search

- The bug is **inside the GPU XC path** (density kernel, accumulate_point, or
  texture fetch), not in the CPU converger, Fortran layer, or AINT.
- It is **triggered by specific density patterns** — molecules with analogous
  electron structure (CH₃OH, H₃PO₄, H₂O₂) are bit-exact at all grids. HCOOH,
  acetic, fosfato trip it.
- The difference between working and broken cases at similar basis sizes
  (CH₃OH 38 bf works vs HCOOH 49 bf broken, CH₃F 36 bf works vs HCOOH 49 bf
  broken) suggests a branch in the density kernel that is only taken for
  certain basis-function distributions or group-size combinations.
- Most likely suspects in `g2g/cuda/kernels/energy.h`:
  1. `fetch_double` texture unpacking (`tex2D<int2>` + `__hiloint2double`) —
     if the texture stride/offset differs from what the kernel expects, int2
     fetches would read the wrong 64 bits.
  2. Warp-shuffle reductions at the end of the density kernel — the float
     path may rely on tolerated rounding, the double path on a specific
     reduction width.
  3. `gpu_accumulate_point` — the XC-evaluation subroutine that takes
     density/gradient and returns energy+potential. Could have an FD-specific
     codepath bug.

### Part 5b — iter=1 single-pass test (fixed initial guess)

Running `nmax=1` with both builds isolates a single Fock build from the
deterministic 1-electron initial guess. The SCF feedback loop is eliminated:
any run-to-run variance or FD-vs-f32 gap measured here is kernel-intrinsic.

| System | Build | Run 1 XC (Ha) | Run 2 XC | Run 3 XC | Spread |
|---|---|---|---|---|---|
| HCOOH | FD | −19.217908 | −19.212546 | −19.215353 | **5.4 mHa** |
| HCOOH | f32 | −19.583087 | (1 run, deterministic elsewhere) | — | — |
| CH₃OH | FD | −12.690899 | −12.690885 | −12.690917 | 32 μHa |
| CH₃OH | f32 | −12.690988 | — | — | — |

Differences:
- HCOOH: FD is **365 mHa below f32** at iter=1, with FD runs scattered over
  **5.4 mHa** among themselves.
- CH₃OH: FD is **90 μHa above f32** at iter=1, with FD runs tight to **32 μHa**.

Two conclusions:

1. **The FD bug fires on iter=1 with a fixed 1-e initial guess on HCOOH** —
   it is not an SCF-path artifact. A single Fock build on a fixed density
   reproduces the error at full magnitude.

2. **FD run-to-run non-determinism on broken systems is 100× larger than on
   working systems** (5 mHa vs 30 μHa). This rules out the OMP/atomicAdd
   reproducibility floor (which gives ~3e-7 Ha on working systems) as the
   only noise source — the broken-system noise is from somewhere else. Most
   likely candidate: an undefined/uninitialized read in the FD kernel path
   that happens to not be memcheck-flagged because the garbage bits happen to
   be valid doubles. memcheck tracks allocations, not semantic correctness of
   read values.

---

## Actionable outputs

1. **For reproducibility work**: document the 3e-7 floor, the 4× rebalance
   amplification, and the 40e-6 rebalance-tipping outlier risk. Going below
   3e-7 requires FULL_DOUBLE — currently unavailable.

2. **For FULL_DOUBLE work**: scope is narrower than the 2026-04-17 note
   suggested. It is **not** a universal regression. The breakage is
   size-dependent and kicks in somewhere between 3 and 34 atoms (or
   equivalently: ~20 vs ~350 basis functions). Bisecting with shrunk
   fosfato inputs is the fastest next step before diving into code.

3. **For `g2g/CLAUDE.md`**: the note says FULL_DOUBLE is "the real fix for
   precision" for the kernel-change-safety section. That's now demonstrably
   wrong at production scales. Either fix FULL_DOUBLE or replace that
   recommendation with "the current float32 noise floor (~1e-6) is inherent
   and cannot be reduced by precision alone."

4. **Float32 GPU is closer to the double-precision oracle than float32 CPU**
   (−4.6e-5 vs +3.4e-5 Ha). This is reassuring — the GPU path is not
   introducing systematic error; it's introducing random noise of the same
   magnitude as CPU discretization error.

---

## Part 6 — Root cause found and fixed (2026-04-19)

**Root cause**: Shared-memory read-after-write race in `gpu_compute_density`
(`g2g/cuda/kernels/energy.h`) and `gpu_compute_density_opened`
(`g2g/cuda/kernels/energy_open.h`). After the outer `bj` loop, threads reuse
the `fj_sh[]`, `fgj_sh[]`, `fh1j_sh[]`, `fh2j_sh[]` shared-memory arrays as
reduction buffers **without a preceding `__syncthreads()`**. Threads that
finish the inner `j`-loop early (e.g. `!valid_thread` paths, `full_block`
early-exit bounds) race ahead and overwrite the cache while other threads are
still reading from it.

**Why FULL_DOUBLE explodes but float32 only degrades**:
`compute-sanitizer racecheck` reports both builds with identical race sites:
190 errors / 681 warnings in float32, comparable count in FD. But 32-bit
shared-memory stores are atomic at word boundaries — a tearing read gets
*either* the old value or the new value of one basis function, an O(ε) error.
64-bit stores split into two 32-bit transactions, so FD reads can observe
partial writes: the low 32 bits of the new value composed with the high 32
bits of the old — a random double with no physical meaning. One tearing read
per point is enough to dominate the XC integral on molecules whose branch
divergence pattern triggers the race (HCOOH, acetic, fosfato).

**Fix**: Added `__syncthreads()` between the outer `bj` loop's final read of
`fj_sh[j]` (energy.h:163, energy_open.h:124/142) and the subsequent
reduction-partials write (energy.h:243, energy_open.h:234). One sync, no
kernel structural change.

### Verification

Rebuilt FULL_DOUBLE with fix (2026-04-19):

| System | Before fix (Ha) | After fix (Ha) | float32 ref (Ha) |
|---|---|---|---|
| HCOOH | −189.446 (128 mHa off) | **−189.573108** (3 runs bit-exact) | −189.573 |
| CH₃OH | −115.602 | −115.602426 | — |
| H₂CO  | −114.394 | −114.394454 | — |
| H₃PO₄ | −643.664 | −643.664358 | — |
| Acetic | (jittery, 34–42 iters) | −228.852682 | — |

HCOOH is now bit-exact over 3 runs (previously had 5.4 mHa run-to-run spread
on iter=1 at fixed guess). The grid-dependence anomaly (FD XC varying 260 mHa
across grids while float32 was invariant) should be gone for the same reason
— that variance was driven by the race outcome, which depends on warp
scheduling and branch patterns per grid.

### Why the race is only now visible

Both builds have had this race since commit `ac87eef0` introduced the
warp-shuffle reductions (which reuse `fj_sh` for the reduction step instead
of the dedicated volatile array). Float32 has been running with 190 race
errors for months — but each race costs O(ε) on a float, which compounds into
the ~2e-6 Ha noise floor we already documented as the "float32 noise floor".
That noise floor is partly fake: some of it was this race, not fundamental
float32 precision loss. A rerun of the reproducibility spread study on the
patched build is the right next step.

### Regression test added

Normal correctness tests cannot catch this class of bug — in float32 the race
produces errors too small to distinguish from regular roundoff. The guards
we added:

1. **`./run_unit.py --sanitize=racecheck`** — upgraded `run_unit.py` to run
   compute-sanitizer's racecheck tool (previously only memcheck). Flags all
   hazards as test failures. Verified by reverting the fix: racecheck reports
   67 errors / 6 warnings on the density kernel, 0 on the fixed build. Same
   detection for the open-shell kernel.

2. **`./run_unit.py --sanitize`** (all tools) — runs memcheck, racecheck,
   initcheck, and synccheck in sequence; any hazard fails the test.

3. **Bit-exact determinism unit tests** in `energy_test.cu` — m=97 pts=4 and
   m=129 pts=3, each run 32× with bit-exact comparison. These are the
   cheapest defense-in-depth; they catch any race that produces visible
   non-determinism (but not the ones that get hidden by float32 store
   atomicity — racecheck is the definitive tool).

Updated `test/CLAUDE.md` to document the racecheck workflow under the
energy/energy_open kernel entries.

---

## Measurements / data retained

All run outputs saved to `/tmp/`:
- Float32 GPU, fos.in (5 runs): `/tmp/fp32_run{1..5}.out`
- Float32 GPU, fos_nocache.in (3): `/tmp/nocache_{1..3}.out`
- Float32 GPU, OMP=1 fos.in (3): `/tmp/omp1_{1..3}.out`
- Float32 GPU, OMP=1 fos_nocache.in (3): `/tmp/omp1nc_{1..3}.out`
- GPU FULL_DOUBLE, fos.in (5): `/tmp/fd_{1..5}.out`
- GPU FULL_DOUBLE, OMP=1 fos_nocache.in (3): `/tmp/fd_omp1nc_{1..3}.out`
- GPU FULL_DOUBLE, fos_noMM.in (3): `/tmp/fd_nomm_{1..3}.out`
- CPU FULL_DOUBLE, fos.in (3 + 1 OMP=1): `/tmp/cpufd_{1..3}.out`, `/tmp/cpufd_omp1.out`
- CPU float32, fos.in (3): `/tmp/cpufp32_{1..3}.out`
- GPU FULL_DOUBLE, agua.in (3): `/tmp/agua_fd_{1..3}.out`
- GPU float32, agua.in (3): `/tmp/agua_fp32_{1..3}.out`
- Sanitizer memcheck of GPU FD fosfato: `/tmp/fdmem.out` (0 errors)

Transient input files created:
- `test/LIO_test/03_fosfatoQMMM/fos_nocache.in` (fgm=0)
- `test/LIO_test/03_fosfatoQMMM/fos_noMM.in` (nsol=0)
