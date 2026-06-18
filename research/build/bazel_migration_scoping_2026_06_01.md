# Bazel Migration — Scoping & Progressive Prototype Plan

Status: **OPEN / scoping only** — no code written. Date: 2026-06-01.

## TL;DR recommendation

**Do not attempt a full big-bang migration. Do a phased, prototype-driven port,
and de-risk the Fortran layer FIRST, not last.** The honest end-state choice is
between two very different projects (see "Two forking decisions"). My
recommendation: a **hybrid** — Bazel for `g2g` (C++/CUDA) where it is clean and
delivers real value, and either a thin-wrapped Make or a scanner-driven custom
Starlark rule for `lioamber` (Fortran). Whether to push Fortran all the way into
Bazel is a go/no-go gated on the Phase-1 prototype below.

The single biggest *win* the user actually asked for ("simplify the Makefiles")
is replacing the `ifeq` option spaghetti across 6 Makefiles with
`config_setting` + `select()`. That alone is worth doing and is low-risk.

The single biggest *risk* is numerical reproducibility: this codebase's SCF
convergence is documented to flip on ULP-level perturbations (thread count, FMA,
`-use_fast_math`, per-file `-O1` vs `-O3`). **Definition of done is bit-identical
e2e energies, not "it links and runs."**

---

## What we're migrating (current state)

Three artifacts, built by a tree of recursive Makefiles:

| Artifact | From | Sources | Hard parts |
|---|---|---|---|
| `libg2g.so` | `g2g/` | 31 `.cpp`, 9 `.cu`, 118 `.h` (~120k LOC) | CUDA separable compile (`-dc`/`-dlink`), gencode fat-binary, version-script export, libxc/libint variants |
| `liblio-g2g.so` | `lioamber/` | 301 `.f90` (~91k LOC), 20 `.mk`, 82 modules | Fortran `.mod` dependency ordering; per-file `-O1`/`-O3`; hand-maintained dep graph |
| `liosolo`, `liomd.x` | `liosolo/` | 2 `.f90` | links both `.so`, `$ORIGIN` rpath |
| `tools/` | 3 sub-makefiles | small Fortran | low priority |

Build-mode matrix (the `ifeq` spaghetti to replace with `select()`):
`cuda=0/1/2/3`, `intel=0/1/2`, `precision=0/1`, `libxc=0/1/2`, `libint=0/1`,
`analytics=0..4`, `fastmath=0/1`, `dbg=0/1`, `aint_mp`, `cpu`, plus per-arch
`smXX=1` flags. Most of these reduce to a handful of `bool_flag` + `config_setting`.

---

## Two forking decisions (resolve these before any code)

### Decision 1 — Hermetic vs. non-hermetic toolchain

The Make build is **aggressively non-hermetic by design** (per root `CLAUDE.md`):

- Requires **bare `nvcc` from PATH = CUDA 12.0**, explicitly NOT
  `$(CUDA_HOME)/bin/nvcc` (13.1 drops Pascal SM 6.x).
- `Makefile.cuda` runs `nvidia-smi --query-gpu=compute_cap` *at build time* to
  pick gencode.
- `-march=native -mtune=native` in both C++ and Fortran flags.

Bazel's headline value (hermeticity, remote cache/exec, reproducibility) fights
all three. Two coherent end-states:

- **(A) Hermetic.** Pin a CUDA 12.0 toolchain, delete the `nvidia-smi` probe, fix
  gencode to SM 8.6 (the dev box) or an explicit arch list, replace `-march=native`
  with an explicit `-march=znver3` (5800X3D). More work; **delivers Bazel's value**
  (cache hits, reproducible binaries, CI parity).
- **(B) Non-hermetic wrapper.** Register the system `gcc`/`gfortran`/`nvcc` as
  local toolchains, keep `nvidia-smi`/`-march=native`. Fast to stand up, but you
  have **mostly rewritten Make in Starlark** — caching is unreliable because inputs
  aren't tracked.

**Recommendation: target (A) hermetic, but reach it incrementally** — start
non-hermetic in the prototype to move fast, then pin toolchains once the graph
builds. State this explicitly so we don't sell Bazel's benefits while the
constraints quietly negate them.

### Decision 2 — Numerical-reproducibility acceptance gate

The codebase-specific landmine. Convergence trajectory is Lyapunov-divergent in
iteration count; documented ULP-level triggers double heme's iters. Make encodes
**per-file optimization levels** (`Makefile.options`): `-O1` for `subm_int3lu`,
`dip`, `subm_intfld`, `subm_int1G/3G`, `subm_intsol*`; `-O3 + OPTIM0I` for SCF/TD
core; `-O3 + OPTIM0P` for math hot paths. A naive uniform `copts=["-O3"]` in Bazel
**will silently shift fosfato/heme/Fe3H2O6 energies**.

Mandated in the plan:
- DoD = **bit-identical e2e energies** vs. the Make build, per mode.
- Reproduce exactly: per-file flags, `-use_fast_math` (default on), link order,
  BLAS/OpenMP thread-cap behavior.
- A CI gate that diffs e2e energies Make-vs-Bazel before any target is "done."

---

## Layer-by-layer port plan

### Mode mapping (do first — this is the user's stated motivation)

Replace `Makefile.translate` / `Makefile.options` `ifeq` blocks with:
```
# //config:BUILD
string_flag(name="cuda", values=["0","1","2"], build_setting_default="2")
config_setting(name="cuda_off",   flag_values={":cuda":"0"})
config_setting(name="cuda_cublas",flag_values={":cuda":"2"})
string_flag(name="libxc", ...)        # 0/1/2
string_flag(name="precision", ...)     # → FULL_DOUBLE
config_setting(name="dbg", ...)        # → -D_DEBUG
```
Then `defines`/`copts` come from `select()`. This is the part that genuinely
"simplifies the Makefiles" — ~6 Makefiles of conditional logic collapse to one
readable `config/` package. **Low risk, high readability payoff. Lead with it.**

### g2g — C++ (CPU path, `cuda=0`)

`cc_library(name="g2g_cpu", srcs=glob(["*.cpp","pointxc/*.cpp","cpu/*.cpp",
"analytic_integral/*.cpp"]), ...)`. Plain and tractable. Reproduce
`-march`, `-fopenmp`, `-DCPU_KERNELS=1`, `-fno-semantic-interposition`, and the
per-mode `-DUSE_LIBXC`/`-DFULL_DOUBLE` defines via `select()`.

### g2g — CUDA (`cuda=1/2`)

Use **`rules_cuda` (bazel-contrib)** — it supports `-dc`/`-dlink` separable
compilation and gencode. Must verify it reproduces:
- the device-link object equivalent to `cuda/lio_gpu.o` (`nvcc -dlink`),
- gencode list (replace the `nvidia-smi` probe with an explicit
  `--@rules_cuda//cuda:archs=sm_86` or a fat list),
- final `.so` link with `-Wl,--version-script=libg2g.map` (export limiting to
  the bridge API) and `$ORIGIN` rpath,
- `-use_fast_math` default-on (reproducibility-sensitive — must be a flag).
- `-lcublas -lcudart -lcuda` and libxc cuda libs.

This is known-tractable; rules_cuda is the right dependency. Budget time for the
version-script + device-link verification, not for "does CUDA build at all."

### lioamber — Fortran (THE LONG POLE — prototype this first)

No turnkey `rules_fortran` exists. The open feasibility question is `.mod`
ordering: a `.mod` file is **both** a compile output and a downstream compile
input, so Bazel needs the per-file DAG.

Two ways to get the DAG:
- **(a) Scanner** (recommended): generate Bazel deps with a fortran module
  scanner (fortdepend / makedepf90 style — `USE x` → needs `x.mod`). Durable;
  survives source churn.
- **(b) Translate the existing graph**: `Makefile.depends` + 20 `.mk` files
  already encode the hand-maintained dep graph — it's an **asset, not a blocker**.
  But it's 301 files / 82 modules to keep in sync by hand. Less durable than (a).

Custom Starlark `fortran_module` rule wrapping `gfortran`, declaring the `.mod` as
an output and consuming upstream `.mod`s via a depset. Must carry per-file
`copts` (the `-O1`/`-O3` split) and `-J/-I` module-dir handling (Bazel sandboxing
vs. gfortran's single `-J` output dir needs care — typically one module dir per
action, collected into a depset).

**Fallback if Fortran-in-Bazel proves too costly:** keep `lioamber` on Make,
expose `liblio-g2g.so` to Bazel as a prebuilt `cc_import`. Hybrid build. This is
a legitimate end-state, not a failure.

### Codegen — `analytic_integral/OS_expand.plx`

Perl generator emitting Obara–Saika term files. **Check first whether outputs are
committed** (they appear to be — treat as source, zero Bazel work) or must be
regenerated (then one `genrule` per term). Don't put codegen on the critical path
if the outputs are already in-tree.

### Tests

- **e2e** (`test/new_tests.py`, `run_travis_test.py`): drive `liosolo` against
  `*.ok` references. Wrap as `sh_test`/`py_test` consuming the Bazel-built
  `liosolo`. This *is* the reproducibility gate.
- **Fortran utests** (`lioamber/utests/`): small `*.x` from single `.f90` →
  trivial Fortran test targets once the Fortran rule exists.
- **GPU kernel unit tests** (`test/unit_tests/kernels/*`): currently committed
  prebuilt ELF binaries (untracked); their sources/build need locating before
  they can become `cuda_test` targets. Lower priority.

---

## Progressive prototype roadmap (de-risk order)

| Phase | Goal | De-risks | Rough effort |
|---|---|---|---|
| **0** | Bazel workspace skeleton; `config/` package with all mode flags as `config_setting`/`select()`; register local (non-hermetic) C++/CUDA/Fortran toolchains | Mode mapping (the stated goal); proves `select()` collapses the `ifeq` mess | 1–2 days |
| **1** | **Fortran de-risk slice**: one leaf module (`constants_mod` or `liosubs_math`) + one consumer that `USE`s it, via a custom `fortran_module` rule. Prove **incremental rebuild** (touch leaf → only dependents recompile) | The feasibility determinant. Go/no-go for full Fortran port | 2–4 days |
| **2** | Build `libg2g.so` CPU-only (`cuda=0`) as `cc_library`; e2e CPU subset bit-identical | C++ path + reproducibility gate harness | 2–3 days |
| **3** | Add CUDA (`cuda=1/2`) via `rules_cuda`; verify device-link, version-script, gencode, `-use_fast_math`; full e2e bit-identical | CUDA-in-Bazel + the export/rpath details | 3–5 days |
| **4** | Full `lioamber` via scanner-generated deps (if Phase 1 went green) OR `cc_import` of Make-built `.so` (hybrid) | The 301-file scale-up | 1–2 wks (full) / 1 day (hybrid) |
| **5** | `liosolo` link + full e2e + utests as Bazel tests; pin hermetic toolchains; CI gate | Hermeticity + CI | 3–5 days |

**Total honest estimate:** ~3–4 weeks for a full hermetic migration **if Phase 1
proves Fortran tractable**; ~1.5–2 weeks for the recommended **hybrid** (Bazel
C++/CUDA, Make Fortran). Phase 0+1 alone (~1 week) produces the go/no-go signal
and already delivers the Makefile-simplification the user wants.

---

## Go / No-Go criteria (decide after Phase 1)

GO to full Fortran-in-Bazel if: the `fortran_module` rule gives correct
incremental rebuilds AND per-file `copts` are expressible AND a scanner produces
the dep DAG without manual per-file maintenance.

Otherwise NO-GO → ship the **hybrid**: Bazel owns `g2g` + `liosolo` + mode flags +
tests; `lioamber` stays on its (already-working, already dependency-ordered) Make
and is consumed as a prebuilt. This still simplifies the build story and captures
most of the value without betting the project on immature Fortran tooling.

## Open questions to confirm before starting
1. Hermetic (A) or wrapper (B) end-state? (Recommend A, reached incrementally.)
2. Are `analytic_integral` term files committed (source) or generated (genrule)?
3. Where do the `test/unit_tests/kernels/*` binaries' sources live / how built?
4. Is full Fortran-in-Bazel a hard requirement, or is the hybrid acceptable?
