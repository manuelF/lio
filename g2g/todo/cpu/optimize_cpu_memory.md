# Optimization: Memory Layout and Transposition

## Summary
The current CPU implementation requires transposing the `function_values` matrix ($P \times M \to M \times P$) explicitly at the end of `compute_functions` using `HostMatrix::transpose` in `g2g/cpu/functions.cpp`. This is a full data copy that incurs significant memory bandwidth overhead.
Additionally, the Density and RMM update loops access data in row-major or column-major patterns that might not be optimal for SIMD or cache (strided access).

## Proposal
1.  **Avoid Explicit Transpose**: Use BLAS (`cblas_dgemm`, `cblas_dsyrk`) which supports `CblasTrans` / `CblasNoTrans` flags directly on the input matrices. Compute directly on the natural layout ($P \times M$ or $M \times P$).
2.  **SoA Optimization**: Ensure `gX`, `gY`... arrays are aligned (32-byte) for AVX.
3.  **Use `mkl_malloc` or `posix_memalign`**: Allocate aligned memory to support aligned loads (`vmovaps`).

## Impact
*   **Memory Bandwidth**: Eliminates the overhead of copying the entire function value matrix.
*   **Performance**: Improves effective memory throughput.
*   **Cache Efficiency**: Avoids polluting the cache with the transposed copy.

## Difficulty Assessment
**Low**

*   **Files to Modify**:
    *   `g2g/cpu/functions.cpp`: Remove the `function_values.transpose()` call at the end of `PointGroupCPU::compute_functions`.
    *   `g2g/cpu/iteration.cpp`: Update `solve_closed` to handle the non-transposed `function_values`. If combining with BLAS optimization, just flip the `CblasTrans` flag.
    *   `g2g/matrix.h` / `g2g/matrix.cpp`: Modify `alloc_data` to use `posix_memalign` or `_mm_malloc`.

*   **Correctness Impact**: **None**, assuming indices are updated correctly.

*   **Dependencies**: Requires `optimize_cpu_blas` to be effective (manual loops are harder to write efficiently for non-transposed data if the access pattern becomes strided).

## Sketch of Changes
1.  **Remove Transpose**:
    *   In `g2g/cpu/functions.cpp`, comment out/remove: `functions.transpose(functions_transposed);`
    *   Ensure `functions` holds the data in $P \times M$ layout (Point-Major).

2.  **Update `solve_closed`**:
    *   If using manual loops: The outer loop is likely over points $p$. Accessing `functions[p*M + i]` is now contiguous (good!). The current code likely expects `functions_transposed[i*P + p]`.
    *   If using BLAS: Update `cblas_dgemm` calls to use the $P \times M$ matrix directly, potentially adjusting `CblasTrans` flags.

3.  **Allocation**:
    *   In `HostMatrix::alloc_data` (`g2g/matrix.cpp`), replace `new T[...]` with:
        ```cpp
        void* ptr;
        posix_memalign(&ptr, 64, size_in_bytes); // 64-byte alignment for AVX-512
        this->data = static_cast<T*>(ptr);
        ```
    *   Update `dealloc_data` to use `free()`.

## Estimations
*   Speedup: 5-10% overall improvement by removing redundant memory operations.