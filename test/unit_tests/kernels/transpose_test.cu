// Unit tests for g2g/cuda/kernels/transpose.h
//
// The transpose kernel is a template __global__ that transposes a 2-D matrix
// stored in row-major order.  Given an input of shape (height × width) it
// produces an output of shape (width × height) such that:
//
//   output[j * height + i] == input[i * width + j]
//
// The kernel tiles the work in 32×32 blocks (TILE_DIM) using 32×8 thread
// blocks (TILE_DIM × BLOCK_ROWS) and pads shared memory by one column to
// avoid bank conflicts.
//
// Test coverage
// -------------
//   1. float  – exact tile multiples (32×32, 64×64, 32×64, 64×32, 96×64…)
//   2. float  – non-multiples of 32, exercising the boundary guards
//   3. float  – degenerate / trivial shapes (1×1, single row, single column)
//   4. float  – large matrix (512×256)
//   5. double – same shapes as float to exercise the template
//   6. Identity property: T(T(A)) == A for non-square matrices

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "test_utils.h"

// Pull the kernel in, matching production usage (it lives inside namespace G2G
// in g2g/cuda/iteration.cu).
namespace G2G {
#include "../../../g2g/cuda/kernels/transpose.h"
}

// ============================================================================
// test_transpose<T>
//
// Allocates host and device buffers, fills the input with a simple linear
// pattern, runs G2G::transpose, then compares the device output against a
// host-computed reference.
//
// Returns true on success, false on the first element mismatch.
// ============================================================================
template <typename T>
static bool test_transpose(int height, int width) {
  const int in_elems = height * width;
  const int out_elems =
      width * height;  // same count, different conceptual layout

  // --- Host buffers ---
  // HostMatrix(width, height)
  std::vector<T> h_in(in_elems);
  std::vector<T> h_out(out_elems, static_cast<T>(0));
  std::vector<T> h_ref(out_elems);

  // Input: A[i][j] = (i * width + j)  — each element is its own flat index.
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < width; ++j) {
      h_in[i * width + j] = static_cast<T>(i * width + j);
    }
  // Reference: A^T[j][i] = A[i][j], output stride = height.
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      h_ref[j * height + i] = h_in[i * width + j];
    }
  }

  // --- Device buffers ---
  T *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, in_elems * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_elems * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, out_elems * sizeof(T)));

  // --- Kernel launch ---
  // Block is always (TILE_DIM, BLOCK_ROWS) = (32, 8).
  // Grid covers the input dimensions; boundary guards inside the kernel
  // handle matrices that are not exact multiples of TILE_DIM.
  dim3 block(TILE_DIM, BLOCK_ROWS);
  dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
            (height + TILE_DIM - 1) / TILE_DIM);

  G2G::transpose<T><<<grid, block>>>(d_out, d_in, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Copy back and verify ---
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_elems * sizeof(T),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_in);
  cudaFree(d_out);

  for (int i = 0; i < width; ++i) {     // output row ∈ [0, width)
    for (int j = 0; j < height; ++j) {  // output col ∈ [0, height)
      if (h_out[i * height + j] != h_ref[i * height + j]) {
        printf("    MISMATCH at output[%d][%d]: expected %.6g  got %.6g\n", i,
               j, (double)h_ref[i * height + j], (double)h_out[i * height + j]);
        return false;
      }
    }
  }
  return true;
}

// ============================================================================
// test_double_transpose<T>
//
// Verifies the identity property: T(T(A)) == A.
// Runs two consecutive transposes on device and checks the round-trip matches
// the original input exactly.
// ============================================================================
template <typename T>
static bool test_double_transpose(int height, int width) {
  const int n = height * width;

  std::vector<T> h_in(n);
  for (int k = 0; k < n; ++k)
    h_in[k] = static_cast<T>(k + 1);  // non-trivial, non-zero values

  // Three device buffers: A (input), B (after first transpose), C (after
  // second)
  T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(T)));
  CUDA_CHECK(
      cudaMemcpy(d_a, h_in.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_b, 0, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(d_c, 0, n * sizeof(T)));

  // First transpose: (height × width) -> (width × height), stored in d_b
  {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
              (height + TILE_DIM - 1) / TILE_DIM);
    G2G::transpose<T><<<grid, block>>>(d_b, d_a, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Second transpose: (width × height) -> (height × width), stored in d_c
  {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((height + TILE_DIM - 1) / TILE_DIM,
              (width + TILE_DIM - 1) / TILE_DIM);
    G2G::transpose<T><<<grid, block>>>(d_c, d_b, height, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<T> h_result(n);
  CUDA_CHECK(
      cudaMemcpy(h_result.data(), d_c, n * sizeof(T), cudaMemcpyDeviceToHost));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  for (int k = 0; k < n; ++k) {
    if (h_result[k] != h_in[k]) {
      printf("    MISMATCH at element %d: expected %.6g  got %.6g\n", k,
             (double)h_in[k], (double)h_result[k]);
      return false;
    }
  }
  return true;
}

// ============================================================================
// main
// ============================================================================
int main() {
  // Print device info so test logs show which GPU was used.
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("transpose kernel");

  // --------------------------------------------------------------------------
  // float tests
  // --------------------------------------------------------------------------
  printf("[ float — exact tile multiples (TILE_DIM = %d) ]\n", TILE_DIM);
  runner.check(test_transpose<float>(32, 32),
               "float  32x32  — single tile, square");
  runner.check(test_transpose<float>(64, 64),
               "float  64x64  — 2x2 tiles, square");
  runner.check(test_transpose<float>(128, 128),
               "float 128x128 — 4x4 tiles, square");
  runner.check(test_transpose<float>(32, 64),
               "float  32x64  — 1x2 tiles, portrait");
  runner.check(test_transpose<float>(64, 32),
               "float  64x32  — 2x1 tiles, landscape");
  runner.check(test_transpose<float>(96, 64), "float  96x64  — 3x2 tiles");
  runner.check(test_transpose<float>(64, 96), "float  64x96  — 2x3 tiles");
  runner.check(test_transpose<float>(128, 256), "float 128x256 — 4x8 tiles");

  printf("\n[ float — non-multiples of %d (boundary guards) ]\n", TILE_DIM);
  runner.check(test_transpose<float>(33, 33),
               "float   33x33  — 1 element past tile boundary");
  runner.check(test_transpose<float>(31, 31),
               "float   31x31  — 1 element short of tile");
  runner.check(test_transpose<float>(50, 70),
               "float   50x70  — arbitrary non-multiples");
  runner.check(test_transpose<float>(100, 200),
               "float 100x200  — multi-tile non-multiples");
  runner.check(test_transpose<float>(65, 97),
               "float   65x97  — straddles 2 tiles in each dim");

  printf("\n[ float — degenerate / edge-case shapes ]\n");
  runner.check(test_transpose<float>(1, 1), "float   1x1   — trivial");
  runner.check(test_transpose<float>(1, 32),
               "float   1x32  — single row, full tile width");
  runner.check(test_transpose<float>(32, 1),
               "float  32x1   — single column, full tile height");
  runner.check(test_transpose<float>(1, 33),
               "float   1x33  — single row, over tile boundary");
  runner.check(test_transpose<float>(3, 5),
               "float   3x5   — very small non-square");
  runner.check(test_transpose<float>(7, 13), "float   7x13  — small odd sizes");

  printf("\n[ float — large matrices ]\n");
  runner.check(test_transpose<float>(512, 512), "float 512x512 — 16x16 tiles");
  runner.check(test_transpose<float>(256, 512), "float 256x512 — 8x16 tiles");
  runner.check(test_transpose<float>(500, 300),
               "float 500x300 — large non-multiples");

  // --------------------------------------------------------------------------
  // double tests (exercises the same template with 8-byte elements)
  // --------------------------------------------------------------------------
  printf("\n[ double ]\n");
  runner.check(test_transpose<double>(32, 32), "double  32x32");
  runner.check(test_transpose<double>(64, 64), "double  64x64");
  runner.check(test_transpose<double>(96, 64), "double  96x64  — non-square");
  runner.check(test_transpose<double>(50, 70),
               "double  50x70  — non-multiples");
  runner.check(test_transpose<double>(3, 5), "double   3x5   — small");
  runner.check(test_transpose<double>(512, 256), "double 512x256 — large");

  // --------------------------------------------------------------------------
  // Identity: T(T(A)) == A
  // --------------------------------------------------------------------------
  printf("\n[ double-transpose identity: T(T(A)) == A ]\n");
  runner.check(test_double_transpose<float>(64, 32), "float  T(T( 64x32))");
  runner.check(test_double_transpose<float>(50, 70), "float  T(T( 50x70))");
  runner.check(test_double_transpose<float>(96, 64), "float  T(T( 96x64))");
  runner.check(test_double_transpose<float>(3, 5), "float  T(T(  3x5 ))");
  runner.check(test_double_transpose<double>(64, 32), "double T(T( 64x32))");
  runner.check(test_double_transpose<double>(50, 70), "double T(T( 50x70))");
  runner.check(test_double_transpose<double>(96, 64), "double T(T( 96x64))");

  return runner.summary();
}
