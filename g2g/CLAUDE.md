# Claude Notes — g2g

Subsystem-specific guidance for AI-assisted development of `g2g`.
See the **root `CLAUDE.md`** for project-wide build commands, environment setup,
running tests, and the CUDA build environment notes (nvcc path, GENCODE_FLAGS).
See `GEMINI.md` for a general architectural overview of this directory.

---

## CUDA Kernel Unit Tests

Low-level tests for individual CUDA kernels live in `test/unit_tests/kernels/`.
They link against nothing in `g2g/` — each test includes the kernel header
directly inside `namespace G2G {}`, matching production usage in `iteration.cu`.

### Adding a new kernel test

1. Drop `<kernel_name>_test.cu` into `test/unit_tests/kernels/`.
2. Write the test:
   - Include kernel headers inside `namespace G2G { #include "../../../g2g/cuda/kernels/<kernel>.h" }`
   - Use `CUDA_CHECK(...)` and `test_utils::TestRunner` from `common/test_utils.h`
3. The shared `kernels/Makefile` picks it up automatically via `$(wildcard *_test.cu)`.

### Build overrides
```bash
# Force a specific GPU architecture
make GENCODE_FLAGS="-gencode arch=compute_80,code=compute_80 \
                    -gencode arch=compute_80,code=sm_80"
# Use a specific nvcc binary
make NVCC=/usr/local/cuda-12.0/bin/nvcc
```

---

## Kernel Notes

### transpose.h (`cuda/kernels/transpose.h`)

**Signature:**
```cpp
template <class T>
__global__ void transpose(T* odata, const T* idata, int width, int height);
```

**Inclusion:** must be inside `namespace G2G {}` (as in `cuda/iteration.cu`).

**Launch config:**
```cpp
dim3 block(TILE_DIM, BLOCK_ROWS);  // (32, 8)
dim3 grid((width + TILE_DIM-1) / TILE_DIM, (height + TILE_DIM-1) / TILE_DIM);
G2G::transpose<T><<<grid, block>>>(d_out, d_in, width, height);
```

**Memory layout:** input `height × width` row-major → output `width × height`
row-major with stride `height`:
```
output[j * height + i] == input[i * width + j]   // for all valid i, j
```

**Resource usage (SM 6.1, CUDA 12.0, `-O0 -G`):**
| Type | Registers | Shared mem |
|---|---|---|
| `float` | 19 | 4224 B (33 × 32 × 4 — the +1 bank-conflict padding) |
| `double` | 20 | 8448 B (33 × 32 × 8) |
