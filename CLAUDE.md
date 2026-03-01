# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is LIO

LIO is a Quantum Mechanical software package based on Density Functional Theory (DFT) and real-time Time-Dependent Density Functional Theory (TD-DFT). It is primarily designed for hybrid QM/MM (Quantum Mechanics/Molecular Mechanics) simulations and runs performance-critical kernels on NVIDIA GPUs via CUDA.

## CUDA Build Environment

**Always use bare `nvcc` (from PATH)**, not `$(CUDA_HOME)/bin/nvcc`.
On this machine `/usr/local/cuda` is managed by `update-alternatives` and
currently symlinks to CUDA 13.1, which **drops Pascal (SM 6.x) support**.
The `nvcc` in PATH resolves to CUDA 12.0 and correctly supports SM 6.1 (GTX 1080).
`g2g/Makefile.cuda` already follows this convention throughout.

GPU arch flags must include **both** PTX and cubin entries so the binary is
usable across driver versions. The detection pattern (from `g2g/Makefile.cuda`):
```makefile
DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap \
                 --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
GENCODE_FLAGS := -gencode arch=compute_$(DETECTED_SM),code=compute_$(DETECTED_SM)
GENCODE_FLAGS += -gencode arch=compute_$(DETECTED_SM),code=sm_$(DETECTED_SM)
```

Current hardware: **GTX 1080, SM 6.1 (Pascal)**, CUDA 12.0 on PATH.

---

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

### CUDA Kernel Unit Tests

Low-level unit tests for individual CUDA kernels live in `test/unit_tests/`.
They compile and run independently of the full library.

```bash
# Build and run all kernel unit tests
cd test/unit_tests && make

# Build only
cd test/unit_tests && make build

# Clean
cd test/unit_tests && make clean
```

To add a test for a new kernel, drop `<name>_test.cu` into `test/unit_tests/kernels/`.
The shared `Makefile` there picks it up automatically. See `g2g/CLAUDE.md` for details.

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

## Code Organization Conventions

### Test directory layout

Three test tiers, each with its own structure:

| Tier | Location | Purpose |
|---|---|---|
| CUDA kernel unit tests | `test/unit_tests/kernels/` | Fast, isolated GPU kernel correctness checks |
| LIO integration tests | `test/LIO_test/NN_<name>/` | Full end-to-end molecule simulations |
| AMBER coupling tests | `test/AMBER_test/<scenario>/` | QM/MM tests requiring AMBER |

### CUDA kernel unit tests — flat layout

All kernel test sources live **directly** in `test/unit_tests/kernels/` — no per-kernel
subfolders. The single shared `Makefile` there auto-discovers every `*_test.cu` via
`$(wildcard *_test.cu)` and builds them all.

**Naming:** `<kernel_name>_test.cu` → binary `<kernel_name>_test`, where `<kernel_name>`
matches the kernel header without `.h` (e.g. `rmm.h` → `rmm_test.cu`).

Adding a new kernel test: drop `<name>_test.cu` in `test/unit_tests/kernels/` — no
Makefile edits needed anywhere. See `g2g/CLAUDE.md` for the include structure.

Shared infrastructure lives in `test/unit_tests/common/`:
- `test_utils.h` — `CUDA_CHECK(...)` macro and `test_utils::TestRunner`
- `Makefile.rules` — reusable toolchain variables for other test suites

### LIO integration tests

Each test lives in `test/LIO_test/NN_<name>/` where `NN` is a zero-padded number
controlling execution order. Required files per test:
- `<name>.in` / `<name>.xyz` — simulation inputs
- `run.sh` — runs liosolo, produces output files
- `check_test.py` — validates outputs against `*.ok` golden references
- `<metric>.ok` — one per checked output (`output.ok`, `forces.ok`, `mulliken.ok`, …)

### AMBER coupling tests

Each scenario under `test/AMBER_test/<scenario>/` uses a three-script convention:
- `test_run*.sh` — runs the simulation
- `test_compare.sh` — diffs output against reference
- `test_clean.sh` — removes generated files

### Python analysis engine

`test/tests_engine/` holds one module per physical observable (`energy.py`, `forces.py`,
`mulliken.py`, `dipole.py`, `fukui.py`, `restart.py`). These are imported by
`test/new_tests.py` and `check_test.py` scripts — they are not standalone executables.

## Git Hooks

A pre-push hook template lives in `hooks/pre-push.py.hook`. When pushing to `master`, it rebuilds from a clean checkout and runs the agua CPU test. To install it:
```bash
cp hooks/pre-push.py.hook .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```
