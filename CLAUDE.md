# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What is LIO

LIO is a Quantum Mechanical software package based on Density Functional Theory (DFT) and real-time Time-Dependent Density Functional Theory (TD-DFT). It is primarily designed for hybrid QM/MM (Quantum Mechanics/Molecular Mechanics) simulations and runs performance-critical kernels on NVIDIA GPUs via CUDA.

## CUDA Build Environment

**Always use bare `nvcc` (from PATH)**, not `$(CUDA_HOME)/bin/nvcc`.
On this machine `/usr/local/cuda` is managed by `update-alternatives` and
currently symlinks to CUDA 13.1, which **drops Pascal (SM 6.x) support**.
The `nvcc` in PATH resolves to CUDA 12.0 and correctly supports SM 6.1 (GTX 1080).

GPU arch flags must include **both** PTX and cubin entries so the binary is
usable across driver versions. The detection pattern is in `g2g/Makefile.cuda`.

Current hardware: **RTX 3080 Ti, SM 8.6 (Ampere)**, CUDA 13.1 on PATH.

---

## Build Commands

```bash
make cuda=1 cpu=0        # Default (GPU kernels only)
make cuda=0 cpu=1        # CPU kernels only
make cuda=1 cpu=1        # Both CPU and GPU
make g2g                 # Just libg2g.so
make liblio              # g2g + lioamber
make liosolo             # Standalone executable
make -C g2g dbg=1        # Debug build (g2g only)
make clean               # Clean everything
```

Key build options:
- `cuda=1|2`: GPU kernels (2 also enables CUBLAS)
- `cpu=1`: CPU OpenMP kernels
- `precision=1`: Full double precision (default is hybrid single/double)
- `aint_mp=1`: float precision for the analytic-integral (Coulomb/QM-MM) kernels (default on)
- `libxc=1|2`: Enable Libxc (1=CPU mode, 2=GPU mode)
- `dbg=1`: Debug symbols + `-D_DEBUG`
- `analytics=0..3`: Profiling/debug verbosity levels

Diagnostic / audit toggles (lioamber, gated — no effect on the default build):
- `lint=1`: surface warning-only flags still being cleaned up
  (`-Warray-temporaries`, `-Wrealloc-lhs`, `-Wmaybe-uninitialized`, `-Wuse-without-only`).
  The correctness guards (`-Wimplicit-interface`, `-Waliasing`, `-fimplicit-none`, …)
  are already on by default. See `lioamber/Makefile.options`.
- `check=1`: runtime `-fcheck=bounds,do,pointer,mem,recursion` + backtrace.
- `check=2`: `-finit-real=snan -ffpe-trap=invalid` — aborts on any uninitialised REAL
  used in arithmetic. Build with `check=N`, then run the e2e suite to surface the bug.

> `-D` / Makefile flag changes don't invalidate the dep tracker (it tracks sources,
> not the Makefile) — run `make clean` after changing a build flag.

## Environment Setup

Before running or testing, source the environment script:
```bash
source liohome.sh
```

## Running Tests

The e2e tests live in `test/LIO_test/<NN_name>/`; each has a `run.sh` (runs
liosolo) and a `check_test.py` (compares against `*.ok` references and exits
non-zero on mismatch).

```bash
make check                                   # From repo root: runs every e2e test
cd test && ./new_tests.py                    # Same, run directly (all e2e tests)
cd test && ./new_tests.py --filter_rx "agua" # Filter by regex on folder name
cd test && ./run_travis_test.py              # Curated CPU-only subset used in CI
make compile                                 # Build matrix sanity check (compile.py)
```

## Architecture

The project compiles two shared libraries used together:

### `g2g/` → `libg2g.so`
The performance-critical C++/CUDA engine. See `g2g/CLAUDE.md` for partition system,
kernel details, profiling data, and optimization history. Key files:
- `partition.h/cpp`: Grid partitioning into `PointGroup` objects
- `init.h/cpp`: Library init and `FortranVars` bridge struct
- `matrix.h/cpp`: `HostMatrix`, `CudaMatrix`, `FortranMatrix` types
- `global_memory_pool.h/cpp`: Custom GPU memory allocator
- `cuda/iteration.cu`: Top-level GPU iteration driver
- `cuda/kernels/`: Individual CUDA kernel headers
- `cpu/iteration.cpp`: CPU counterpart
- `analytic_integral/`: Coulomb/QM-MM integrals

### `lioamber/` → `liblio-g2g.so`
Fortran 90 QM logic layer. Calls into `libg2g.so` for numerical work. Key files:
- `liomain.f90`: Entry point; drives SCF or Ehrenfest dynamics
- `init_lio.f90`: Input parsing, initialization
- `liomods/garcha_mod.f`: Global simulation state module
- `liosubs/`: Utility subroutines
- `faint_cpu/`: Fortran-side CPU integral routines
- `ehrensubs/`: Real-time TD-DFT subroutines

### `liosolo/liosolo` executable
Standalone Fortran driver for testing without AMBER/GROMACS.

### Language interoperability
Fortran calls C++ via `extern "C"` bindings with trailing underscores (e.g., `g2g_init_`).
`FortranVars` in `g2g/init.h` is the shared state. `FortranMatrix` handles column-major layout.

## Research & Optimization Knowledge Base

`research/` contains all optimization research, profiling analysis, bug investigations,
and technical evaluations. This is the project's
institutional memory for performance work.

### How to navigate it

There is **no top-level `research/INDEX.md`** — navigation starts at the per-area
sub-indexes listed in the table below.

1. **Open the relevant `research/<area>/INDEX.md`** — each lists every file in that
   area with status (`DONE` / `OPEN` / `REJECTED` / `DEAD` / `PLAN`), an impact rating,
   and a one-line summary. Read only the sub-index, not every file.
2. **Read individual files only when you need the details** for a specific optimization
   you're about to implement or a constraint you need to understand.

### When to consult research/

- **Before any SCF / convergence change**: read `research/convergence/INDEX.md`. The
  float32-XC noise + DIIS ulp-sensitivity make iter-count the dominant lever, and
  **heme iter-count is Lyapunov-chaotic — judge every change on the median across
  OMP={1,4,6,8}, not the lucky default.** The master catalogue + resume point is
  `research/convergence/scf_iteration_reduction_strategies_2026_06_14.md`.
- **Before re-investigating a closed topic**: each sub-index flags dead ends inline
  with `REJECTED` / `DEAD` status and the data that killed them. Several approaches
  (Kahan summation, `__ldg`, level shifting, TF32/tensor cores for the density path,
  CUDA Graphs, per-TD-step int3lu overlap, partition-geometry/mega-kernel) have been
  tested and ruled out — check before retrying.
- **No-toggle rule (project owner)**: an optimization ships unconditionally or stays
  out; do not add `LIO_*` env tuning levers for new work.
- **When writing new research**: add the file to the appropriate `research/<area>/`
  folder and update that area's `INDEX.md` with status and summary.

### Area quick reference

| Sub-index | When to read |
|-----------|--------------|
| `research/gpu/INDEX.md` | Modifying any CUDA kernel (density, RMM, forces) |
| `research/cpu/INDEX.md` | Modifying the CPU code path / faint_cpu / diag |
| `research/convergence/INDEX.md` | Touching anything that feeds the SCF loop (DIIS, guess, criterion) |
| `research/fortran/INDEX.md` | Working in `lioamber/` (overlap, BLAS, interface-lint) |
| `research/tddft/INDEX.md` | TD-DFT / Ehrenfest / Magnus propagation |
| `research/build/INDEX.md` | Build system / Bazel-migration scoping |

## Code Style

- **C++**: Google C++ style via `.clang-format` in `g2g/`. Run `clang-format -i` before committing.
- **Precision**: Kernels templated on `scalar_type` (`float`/`double`) via `FULL_DOUBLE` macro.
- **CUDA block sizes**: Constants in `g2g/common.h` — must be multiples of 16.
- **Fortran**: `implicit none`, allocatable arrays, `garcha_mod` for global state.

## Git Hooks

Pre-push hook template: `hooks/pre-push.py.hook`. Rebuilds and runs agua CPU test on push to `master`.
```bash
cp hooks/pre-push.py.hook .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
