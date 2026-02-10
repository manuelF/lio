# Optimization: Move Semantics for HostMatrix

## Summary
The `HostMatrix` class currently implements copy constructors and copy assignment operators that perform deep copies of the underlying data (using `memcpy` or `cudaMemcpy`).
It lacks move constructors and move assignment operators (C++11).
Consequently, returning `HostMatrix` from functions or using them in `std::vector` resize operations can trigger expensive and unnecessary deep copies of large memory buffers.

## Proposal
Implement move semantics for `HostMatrix<T>` and `CudaMatrix<T>`.
1.  **Move Constructor**: `HostMatrix(HostMatrix&& other)`. Transfer ownership of `other.data` pointer to `this`. Set `other.data` to `nullptr` and `other.width/height` to 0.
2.  **Move Assignment**: `HostMatrix& operator=(HostMatrix&& other)`. Deallocate current `data` (if any), transfer ownership from `other`, and zero out `other`.

## Impact
*   **Performance**: Eliminates expensive memory copies when passing matrices by value or resizing vectors containing matrices.
*   **Efficiency**: Reduces memory churn.

## Difficulty Assessment
**Low**

*   **Files to Modify**:
    *   `g2g/matrix.h`: Add method declarations to `HostMatrix` and `CudaMatrix` classes.
    *   `g2g/matrix.cpp`: Implement the logic.

*   **Correctness Impact**: **Low**. Standard C++ idiom. Just ensure `other` is left in a valid empty state to prevent double-free destructor issues.

## Sketch of Changes
In `matrix.h` and `matrix.cpp`:
```cpp
// Move Constructor
template<class T>
HostMatrix<T>::HostMatrix(HostMatrix<T>&& other) noexcept : Matrix<T>() {
    this->data = other.data;
    this->width = other.width;
    this->height = other.height;
    this->pinned = other.pinned;
    
    other.data = nullptr;
    other.width = 0;
    other.height = 0;
}

// Move Assignment
template<class T>
HostMatrix<T>& HostMatrix<T>::operator=(HostMatrix<T>&& other) noexcept {
    if (this != &other) {
        this->deallocate(); // Important: free current resources!
        this->data = other.data;
        this->width = other.width;
        this->height = other.height;
        this->pinned = other.pinned;
        
        other.data = nullptr;
        other.width = 0;
        other.height = 0;
    }
    return *this;
}
```
*   Repeat for `CudaMatrix`.
*   Ensure `virtual` destructors in `Matrix<T>` base class handle null pointers gracefully.

## Estimations
*   Speedup: Noticeable reduction in overhead for vector resizing and factory functions.