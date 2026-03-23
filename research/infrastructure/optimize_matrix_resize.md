# Optimization: Avoid Reallocation on HostMatrix::resize

## Summary
The current implementation of `HostMatrix::resize` (`g2g/matrix.cpp`) always deallocates existing memory (`dealloc_data()`) and allocates new memory (`alloc_data()`) whenever the dimensions change, even if the new size is smaller than the current capacity.
This leads to significant memory allocator churn (`malloc`/`free` or `cudaMallocHost`/`cudaFreeHost`) inside iterative loops where matrices might be resized frequently (e.g., dynamic groups).

## Proposal
Implement a capacity-aware resize mechanism.
1.  **Track Capacity**: Add `size_t capacity` member to `HostMatrix`.
2.  **Smart Resize**:
    *   If `new_size <= capacity`, update `width` and `height` without reallocating.
    *   Only reallocate if `new_size > capacity`.
    *   (Optional) Implement `shrink_to_fit` if memory needs to be released.

## Impact
*   **Performance**: Eliminates allocation overhead for frequently resized matrices.
*   **Fragmentation**: Reduces heap fragmentation.

## Difficulty Assessment
**Low**

*   **Files to Modify**:
    *   `g2g/matrix.h`: Add `size_t _capacity` member.
    *   `g2g/matrix.cpp`: Update `alloc_data`, `resize`, `operator=`.

*   **Correctness Impact**: **Low**. Must ensure that data from previous usage doesn't leak logically (though `resize` usually doesn't guarantee preserving data, standard `vector::resize` does preserve common elements, but here `HostMatrix::resize` likely invalidates content. We should clarify if it's `resize` vs `reserve`). The current `HostMatrix::resize` seems to be destructive.

## Sketch of Changes
1.  **Modify `HostMatrix`**:
    *   Add `size_t _capacity = 0;` protected member.
    *   In `alloc_data()`: `_capacity = width * height * sizeof(T); ...`
    *   In `resize(w, h)`:
        ```cpp
        size_t needed_bytes = w * h * sizeof(T);
        if (needed_bytes > _capacity) {
            deallocate(); // Reset capacity inside here? Or handle manually.
            this->width = w; this->height = h;
            alloc_data(); // Sets _capacity
        } else {
            this->width = w; this->height = h;
            // Data remains valid pointer, content undefined implies we don't need to zero it? 
            // If safety needed, memset(data, 0, needed_bytes).
        }
        ```

## Estimations
*   Speedup: Noticeable reduction in overhead for iterative solvers with variable group sizes.