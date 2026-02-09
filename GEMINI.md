# LIO Project Overview

LIO is a Quantum Mechanical software package based on Density Functional Theory (DFT) and real-time Time-Dependent Density Functional Theory (TD-DFT). It is designed for high-performance simulations, particularly hybrid QM/MM (Quantum Mechanics/Molecular Mechanics) simulations, with significant acceleration provided by GPU kernels using CUDA.

## Core Technologies
- **Languages:** C++ (core kernels), Fortran 90 (high-level QM logic and interface), CUDA (GPU acceleration), Python (testing and utilities).
- **Libraries:** LAPACK or Intel MKL, NVIDIA CUDA, Libxc (optional).
- **Build System:** GNU Make.

## Project Structure
- `g2g/`: Contains the core implementation of DFT/TD-DFT kernels in C++ and CUDA. It produces `libg2g.so`.
- `lioamber/`: Contains the Fortran 90 code that implements the QM logic and provides an interface for integration with molecular dynamics packages like AMBER. It produces `liblio-g2g.so`.
- `liosolo/`: A standalone driver for LIO, producing the `liosolo` executable.
- `test/`: A comprehensive test suite organized by functionality (LIO standalone, AMBER integration, Libxc).
- `dat/`: Basis sets, ECP (Effective Core Potentials) data, and conversion scripts.
- `tools/`: Miscellaneous utility tools.
- `docs/`: Documentation and manuals.

## Environment Setup
The script `liohome.sh` in the root directory should be sourced to set up the necessary environment variables:
```bash
source liohome.sh
```
This sets `LIOHOME`, and updates `PATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH`.

## Building the Project
Compilation is managed via the root `Makefile`. Several options can be passed to `make`:

```bash
make [cpu=1] [cuda=1|2] [intel=1|2] [precision=1] [libxc=1|2]
```

- `cpu=1`: Compile CPU kernels.
- `cuda=1`: Compile GPU kernels (default). `cuda=2` also uses CUBLAS.
- `intel=1`: Use Intel compilers. `intel=2` also uses Intel MKL.
- `precision=1`: Use double precision (default is hybrid).
- `libxc=1|2`: Enable Libxc support (1 for CPU, 2 for GPU mode).

Key build targets:
- `make all`: Builds everything (g2g, lioamber, liosolo, tools).
- `make liblio`: Builds `libg2g.so` and `liblio-g2g.so`.
- `make liosolo`: Builds the standalone executable.

## Running Tests
Tests can be executed from the root using:
```bash
make check
```
Alternatively, navigate to `test/LIO_test/` and run the `run.sh` script within any of the test subdirectories.

## Development Conventions
- **Code Style:** The C++ code in `g2g/` follows a style defined in `.clang-format`.
- **Modularity:** The project separates performance-critical kernels (C++/CUDA in `g2g/`) from the high-level scientific logic (Fortran in `lioamber/`).
- **Integration:** LIO is designed to be linked as a dynamic library into other simulation packages (AMBER, GROMACS).

## Key Files
- `liohome.sh`: Essential environment configuration.
- `liosolo/liosolo.f90`: Entry point for the standalone version.
- `lioamber/liomain.f90`: Main entry points for the LIO library.
- `g2g/init.cpp`: Initialization logic for the C++ backend.
- `README.md`: Detailed installation and usage instructions.
