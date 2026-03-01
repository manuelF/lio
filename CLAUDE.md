# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is LIO

LIO is a Quantum Mechanical software package based on Density Functional Theory (DFT) and real-time Time-Dependent Density Functional Theory (TD-DFT). It is primarily designed for hybrid QM/MM (Quantum Mechanics/Molecular Mechanics) simulations and runs performance-critical kernels on NVIDIA GPUs via CUDA.

## Build Commands

```bash
# Default build (GPU kernels, no CPU kernels)
make cuda=1 cpu=0

# Build with CPU kernels only
make cuda=0 cpu=1

# Build with both CPU and GPU
make cuda=1 cpu=1

# Build just the g2g shared library
make g2g

# Build the full library stack (g2g + lioamber)
make liblio

# Build the standalone executable
make liosolo

# Debug build (g2g only, adds -g -ggdb)
make -C g2g dbg=1

# Clean everything
make clean
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
This sets `LIOHOME`, `PATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH` to include `g2g/` and `lioamber/`.

## Running Tests

```bash
# Run all LIO standalone tests
cd test && ./new_tests.py

# Run a filtered subset (uses Python regex)
cd test && ./new_tests.py --filter_rx "00_agua"

# Run a single test manually
cd test/LIO_test/00_agua && ./run.sh

# Run via make from root
make check
```

Each test directory under `test/LIO_test/` contains a `run.sh` script and a `check_test.py` that validates the output against `.ok` reference files.

## Architecture

The project compiles two shared libraries used together:

### `g2g/` → `libg2g.so`
The performance-critical C++/CUDA engine. All computationally intensive DFT/TD-DFT kernels live here. Key files:
- `partition.h/cpp`: Grid partitioning; distributes integration points into `PointGroup` objects for CPU/GPU execution
- `init.h/cpp`: Library initialization and the `FortranVars` struct — the central data-sharing bridge between Fortran and C++
- `matrix.h/cpp`: Custom matrix types: `HostMatrix`, `CudaMatrix`, `FortranMatrix` (column-major)
- `global_memory_pool.h/cpp`: Custom GPU memory allocator to avoid repeated `cudaMalloc`/`cudaFree`
- `cuda/iteration.cu`: Top-level GPU iteration driver; includes all kernel headers
- `cuda/kernels/`: Individual CUDA kernel implementations (energy, force, rmm, weight, functions, transpose, etc.)
- `cpu/iteration.cpp`: CPU counterpart
- `pointxc/`: Point-wise exchange-correlation functional implementations (LDA, GGA, closed/open shell)
- `analytic_integral/`: Analytical integral code for Coulomb/QM-MM

### `lioamber/` → `liblio-g2g.so`
The Fortran 90 QM logic layer. It calls into `libg2g.so` for numerical work. Key files:
- `liomain.f90`: Central entry point; drives SCF or Ehrenfest dynamics
- `init_lio.f90`: Reads input, initializes the system
- `liomods/garcha_mod.f`: The main Fortran `module` holding global simulation state (basis, matrices, flags)
- `liosubs/`: Utility subroutines (I/O, math, error handling)
- `faint_cpu/`: Fortran-side CPU integral routines
- `ehrensubs/`: Real-time TD-DFT (Ehrenfest dynamics) subroutines

### `liosolo/` → `liosolo` executable
A standalone Fortran driver that calls into `liblio-g2g.so`. Used for testing without AMBER/GROMACS.

### Language interoperability
Fortran calls C++ functions via `extern "C"` bindings with trailing underscores (e.g., `g2g_init_`, `g2g_timer_sum_start_`). The `FortranVars` struct in `g2g/init.h` is populated at initialization time and acts as the shared state between both layers. `FortranMatrix` wrappers handle column-major ↔ row-major differences.

## Code Style

- **C++**: Follow Google C++ style via `.clang-format` in `g2g/`. Run `clang-format -i` before committing C++ changes.
- **Precision**: Core classes and kernels are templated on `scalar_type` (`float` or `double`) controlled by the `FULL_DOUBLE` macro.
- **CUDA block sizes**: Constants like `FUNCTIONS_BLOCK_SIZE`, `WEIGHT_BLOCK_SIZE`, etc. are defined in `g2g/common.h` — must be multiples of 16.
- **Fortran**: Uses `implicit none`, allocatable arrays, and `garcha_mod` for global state.

## Git Hooks

A pre-push hook template lives in `hooks/pre-push.py.hook`. When pushing to `master`, it rebuilds from a clean checkout and runs the agua CPU test. To install it:
```bash
cp hooks/pre-push.py.hook .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```
