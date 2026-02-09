# G2G CPU Kernels

This directory contains the C++ implementations of the core computational kernels for the LIO package, specifically optimized for CPU execution using OpenMP. These kernels provide a fallback or alternative to the GPU-accelerated (CUDA) versions.

## Project Overview

The CPU kernels are responsible for evaluating basis functions, computing numerical integration weights, and solving the exchange-correlation (XC) part of the DFT equations. They are designed to be efficient and parallelized across multiple CPU cores.

### Core Technologies
- **Languages:** C++ (C++11/14).
- **Parallelism:** OpenMP for multi-core acceleration.
- **Interoperability:** Heavily interacts with the Fortran backend through `FortranVars` and supports `Libxc` for exchange-correlation functionals.
- **Precision:** Supports both single and double precision via template specialization (`scalar_type`).

## Key Components

### 1. Basis Function Evaluation (`functions.cpp`)
- Implements `PointGroupCPU::compute_functions`.
- Evaluates s, p, and d-type Gaussian basis functions at specific grid points.
- Computes first (gradient) and second (hessian) derivatives of basis functions, which are essential for GGA functionals and force calculations.

### 2. Numerical Integration Solvers (`iteration.cpp`)
- Implements `solve_closed` and `solve_opened` for closed-shell and open-shell systems.
- Performs the integration of the density matrix to obtain the XC energy and potential.
- Integrates with local LDA/GGA implementations and the `Libxc` library.
- Handles the calculation of atomic forces on the CPU.

### 3. Grid Weighting (`weight.cpp`)
- Implements `PointGroupCPU::compute_weights`.
- Calculates Becke partitioning weights for the numerical integration grid, ensuring proper distribution of contributions from different atoms.

### 4. Interface Definitions (`exchnum.h`)
- Defines the low-level `cpu_compute_density_forces` interface.

## Building and Development

### Compilation
The CPU kernels are compiled as part of the main `g2g` library. Ensure the `cpu=1` flag is used with `make` in the project root or the `g2g` directory:
```bash
make cpu=1
```

### Development Conventions
- **Parallelism:** Use `#pragma omp parallel for` for loops over grid points or basis functions. Ensure thread safety when accumulating results (e.g., using `reduction`).
- **Memory Management:** Utilize `HostMatrix` for CPU-side data storage. Avoid frequent allocations inside hot loops; use `resize` and `zero` where possible.
- **Precision:** Always use the `scalar_type` template parameter to ensure compatibility with both single and double precision builds.
- **Optimization:** Prefer vectorized operations and minimize branching in the inner loops of `compute_functions` and `solve_closed/opened`.
