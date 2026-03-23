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

Current hardware: **GTX 1080, SM 6.1 (Pascal)**, CUDA 12.0 on PATH.

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
- `intel=1|2`: Use Intel compilers (2 also uses MKL)
- `precision=1`: Full double precision (default is hybrid single/double)
- `libxc=1|2`: Enable Libxc (1=CPU mode, 2=GPU mode)
- `dbg=1`: Debug symbols + `-D_DEBUG`
- `analytics=0..3`: Profiling/debug verbosity levels

## Environment Setup

Before running or testing, source the environment script:
```bash
source liohome.sh
```

## Running Tests

See `test/CLAUDE.md` for the full test guide including unit tests, e2e tests,
filtering, and which tests to run for each file changed.

```bash
cd test && ./run_tests.py              # Run everything
cd test && ./run_unit.py               # Unit tests only (~20s)
cd test && ./run_e2e.py                # E2E integration tests (~60s)
cd test && ./run_tests.py --filter_rx "energy"  # Filter by regex
make check                             # From repo root
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

### `liosolo/` → `liosolo` executable
Standalone Fortran driver for testing without AMBER/GROMACS.

### Language interoperability
Fortran calls C++ via `extern "C"` bindings with trailing underscores (e.g., `g2g_init_`).
`FortranVars` in `g2g/init.h` is the shared state. `FortranMatrix` handles column-major layout.

## Research & Optimization Knowledge Base

All optimization research, profiling analysis, and technical investigations
live in `research/`. See [`research/INDEX.md`](research/INDEX.md) for the
central index with status tracking and priority rankings.

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
