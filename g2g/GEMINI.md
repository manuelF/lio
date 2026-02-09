# G2G: LIO High-Performance Kernels

`g2g` is the core computational engine of the LIO Quantum Mechanical software package. It provides high-performance C++ and CUDA implementations of Density Functional Theory (DFT) and real-time Time-Dependent Density Functional Theory (TD-DFT) kernels.

## Project Overview

The primary goal of `g2g` is to provide efficient, GPU-accelerated kernels for evaluating energy, forces, and other properties in QM and QM/MM simulations. It is designed to be linked as a dynamic library (`libg2g.so`) and called from high-level logic (primarily Fortran 90 in `lioamber`).

### Core Technologies
- **Languages:** C++ (standard C++11/14), CUDA (for GPU acceleration), OpenMP (for CPU parallelism).
- **Libraries:** CUDA Toolkit, NVIDIA CUBLAS (optional), Intel MKL (optional), Libxc (optional integration).
- **Architecture:** Specialized for hybrid CPU/GPU execution with custom memory and matrix management.

## Project Structure

- `cuda/`: CUDA source files and kernels for GPU-based computations.
- `cpu/`: C++ source files for CPU-based computations.
- `analytic_integral/`: Implementation of analytical integrals (C++ and CUDA).
- `pointxc/`: Point-wise Exchange-Correlation functional implementations.
- `libxc/`: Integration layer for the Libxc library.
- `datatypes/`: Core data structures and primitive types.
- `matrix.h/cpp`: Custom matrix library supporting Host, CUDA, and Fortran-interfaced matrices.
- `partition.h/cpp`: Logic for grid partitioning and work distribution.
- `init.h/cpp`: Library initialization and inter-language variable mapping (`FortranVars`).
- `global_memory_pool.h/cpp`: Custom memory management for efficient GPU resource utilization.

## Building and Running

The build system is based on GNU Make. The main `Makefile` in this directory handles the compilation of the `libg2g.so` library.

### Key Build Options
- `cuda=1|2`: Enable CUDA kernels (1 for basic, 2 for CUBLAS).
- `cpu=1`: Enable CPU kernels.
- `intel=1|2`: Use Intel compilers and MKL.
- `libxc=1|2`: Enable Libxc support (1 for CPU, 2 for GPU mode).
- `full_double=1`: Use double precision (default is hybrid/single precision).
- `dbg=1`: Enable debug symbols and extra logging.

### Commands
- `make`: Builds `libg2g.so` using default options.
- `make clean`: Removes object files and the shared library.
- `make depend`: Regenerates header dependencies.

## Development Conventions

- **Code Style:** C++ code follows the project's `.clang-format` specification.
- **Precision:** Most core classes and kernels are templated on `scalar_type` to support both single and double precision based on the `FULL_DOUBLE` macro.
- **Memory Management:** Use `GlobalMemoryPool` for allocating GPU memory to avoid expensive `cudaMalloc`/`cudaFree` calls during execution.
- **Matrices:** Use `G2G::HostMatrix`, `G2G::CudaMatrix`, and `G2G::FortranMatrix` for data handling. Note that `FortranMatrix` assumes column-major ordering as provided by the Fortran caller.
- **Error Handling:** Use standard C++ exceptions and `CUDA_CHECK` style macros for CUDA API calls.
- **Parallelism:** CPU loops should be parallelized with OpenMP where appropriate. GPU kernels should be optimized for coalesced access and minimal register pressure.
