# G2G: CUDA Kernels for LIO

This directory contains the GPU-accelerated implementation of the LIO Quantum Mechanical kernels. These kernels are written in CUDA C++ and are designed to perform high-performance evaluations of energy, forces, and density on NVIDIA GPUs.

## Directory Overview

- `iteration.cu`: The primary implementation file for the GPU-based QM logic. It defines the `PointGroupGPU` class methods, which orchestrate the execution of CUDA kernels for both closed-shell and open-shell systems.
- `gpu_variables.h`: Defines `__device__ __constant__` variables used across kernels, such as atomic positions and normalization factors.
- `cuda_extra.h`: Likely contains utility functions and macros for CUDA (e.g., error checking).
- `kernels/`: A subdirectory containing the individual CUDA kernels, each specialized for a specific part of the QM calculation:
    - `accumulate_point.h`: Kernels for accumulating density and energy at grid points.
    - `energy.h` / `energy_open.h`: Kernels for computing density and energy contributions.
    - `energy_derivs.h`: Kernels for computing derivatives of the density/energy.
    - `force.h`: Kernels for computing atomic forces.
    - `functions.h`: Kernels for evaluating basis functions and their gradients/hessians on the grid.
    - `rmm.h`: Kernels for updating the Reduced Density Matrix (RMM).
    - `transpose.h`: Utility kernels for matrix transposition to optimize memory access patterns.
    - `weight.h`: Kernels for computing quadrature weights.
    - `functions.h`: Implementation of basis function evaluations.

## Architecture and Design

- **Precision:** Most kernels and host-side orchestration logic are templated on `scalar_type` to support both single and double precision (controlled by the `FULL_DOUBLE` macro).
- **Memory Management:**
    - Uses `GlobalMemoryPool` for efficient GPU memory allocation.
    - Employs `texture` memory (e.g., `rmm_input_gpu_tex`) for certain input matrices to leverage hardware-level caching.
    - Uses `__constant__` memory for frequently accessed global data like `gpu_atom_positions`.
- **Integration:** Supports integration with `Libxc` for exchange-correlation functional evaluations, with both CPU and GPU fallback options.
- **Parallelism:** Kernels are optimized for coalesced memory access and use shared memory (`__shared__`) to reduce global memory bandwidth requirements.

## Building

These files are compiled as part of the `libg2g.so` library. The build is typically triggered from the root or `g2g/` directory using:

```bash
make cuda=1 # or cuda=2 for CUBLAS support
```

## Key Symbols

- `G2G::PointGroupGPU<scalar_type>::solve_closed()`: Entry point for closed-shell GPU calculations.
- `G2G::PointGroupGPU<scalar_type>::solve_opened()`: Entry point for open-shell GPU calculations.
- `gpu_compute_functions()`: Kernel for basis function evaluation.
- `gpu_compute_density()`: Kernel for density matrix contraction.
- `gpu_compute_forces()`: Kernel for force evaluation.
