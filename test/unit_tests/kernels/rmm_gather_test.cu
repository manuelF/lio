// Unit tests for g2g/cuda/kernels/rmm_gather.h
//
// gpu_gather_rmm<scalar_type>:
//   For each index k in [0, n_indexes):
//     val = (scalar_type)global_rmm[bigs[k]]
//     local_rmm[cols[k] * rmm_width + rows[k]] = val   (lower triangle)
//     local_rmm[rows[k] * rmm_width + cols[k]] = val   (upper triangle, mirror)
//
// Test coverage
// -------------
//   1. m=1: trivial (single element, row==col)
//   2. m=2: hand-verify lower+upper triangle fill
//   3. m=4: compare GPU gather vs CPU reference gather
//   4. m=30: realistic group size, verify symmetry
//   5. m=4: verify that padding (COALESCED_DIMENSION) stays zero

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC   0

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "test_utils.h"
#include "../../../g2g/common.h"               // COALESCED_DIMENSION
#include "../../../g2g/matrix.h"
#include "../../../g2g/scalar_vector_types.h"

namespace G2G {
#include "../../../g2g/cuda/kernels/rmm_gather.h"
}

// ---------------------------------------------------------------------------
// CPU reference: build packed upper-triangular global RMM index
// big_index(i,j) = i * M - i*(i-1)/2 + (j - i)   where i <= j
// ---------------------------------------------------------------------------
static uint packed_index(uint i, uint j, uint M) {
  if (i > j) { uint t = i; i = j; j = t; }
  return i * M - (i * (i - 1)) / 2 + (j - i);
}

// CPU reference gather: mirrors get_rmm_input + symmetrize
static std::vector<float> ref_gather(
    const std::vector<double>& global_rmm,
    const std::vector<uint>& bigs,
    const std::vector<uint>& rows,
    const std::vector<uint>& cols,
    uint group_m) {
  uint rmm_width = COALESCED_DIMENSION(group_m);
  uint height = group_m + DENSITY_BLOCK_SIZE;
  std::vector<float> local(rmm_width * height, 0.f);
  for (size_t k = 0; k < bigs.size(); k++) {
    float val = (float)global_rmm[bigs[k]];
    uint r = rows[k], c = cols[k];
    local[c * rmm_width + r] = val;  // lower triangle
    if (r != c) local[r * rmm_width + c] = val;  // upper triangle
  }
  return local;
}

// ---------------------------------------------------------------------------
// Run GPU gather kernel
// ---------------------------------------------------------------------------
static std::vector<float> run_gather(
    const std::vector<double>& global_rmm,
    const std::vector<uint>& bigs,
    const std::vector<uint>& rows,
    const std::vector<uint>& cols,
    uint group_m) {
  uint rmm_width = COALESCED_DIMENSION(group_m);
  uint height = group_m + DENSITY_BLOCK_SIZE;
  uint n_indexes = (uint)bigs.size();
  uint out_size = rmm_width * height;

  double* d_global;
  uint *d_bigs, *d_rows, *d_cols;
  float* d_local;

  CUDA_CHECK(cudaMalloc(&d_global, global_rmm.size() * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_bigs,   n_indexes * sizeof(uint)));
  CUDA_CHECK(cudaMalloc(&d_rows,   n_indexes * sizeof(uint)));
  CUDA_CHECK(cudaMalloc(&d_cols,   n_indexes * sizeof(uint)));
  CUDA_CHECK(cudaMalloc(&d_local,  out_size * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_global, global_rmm.data(),
                        global_rmm.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bigs, bigs.data(), n_indexes * sizeof(uint),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rows, rows.data(), n_indexes * sizeof(uint),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cols, cols.data(), n_indexes * sizeof(uint),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_local, 0, out_size * sizeof(float)));

  dim3 block(256);
  dim3 grid((n_indexes + 255) / 256);
  G2G::gpu_gather_rmm<float><<<grid, block>>>(
      d_global, d_bigs, d_rows, d_cols, d_local, n_indexes, rmm_width);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_local(out_size);
  CUDA_CHECK(cudaMemcpy(h_local.data(), d_local, out_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_global);
  cudaFree(d_bigs); cudaFree(d_rows); cudaFree(d_cols);
  cudaFree(d_local);
  return h_local;
}

// Build index arrays for a group that includes ALL M global functions
// (identity mapping: local func i = global func i).
static void build_full_indexes(uint M,
                               std::vector<uint>& bigs,
                               std::vector<uint>& rows,
                               std::vector<uint>& cols) {
  bigs.clear(); rows.clear(); cols.clear();
  for (uint i = 0; i < M; i++) {
    for (uint j = i; j < M; j++) {
      uint big = packed_index(i, j, M);
      // In compute_indexes: if (ii > jj) swap(ii,jj)
      // With identity mapping and i <= j: row=i, col=j
      rows.push_back(i);
      cols.push_back(j);
      bigs.push_back(big);
    }
  }
}

// ============================================================================
int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_gather_rmm kernel");

  const float tol = 1e-6f;

  // --- 1. m=1: trivial ---
  // Global RMM has 1 element: R[0][0] = 3.5
  // bigs=[0], rows=[0], cols=[0]
  // local_rmm[0 * rmm_width + 0] = 3.5
  {
    std::vector<double> global = {3.5};
    std::vector<uint> bigs = {0}, rows = {0}, cols = {0};
    auto got = run_gather(global, bigs, rows, cols, 1);
    runner.check(fabsf(got[0] - 3.5f) < tol, "m=1 trivial");
  }

  // --- 2. m=2: hand-verify ---
  // Global packed: R[0][0]=1.0, R[0][1]=2.0, R[1][1]=3.0
  // pack index: (0,0)=0, (0,1)=1, (1,1)=2
  // group_m=2, rmm_width = COALESCED_DIMENSION(2) = 32+2-2%32 = 32
  // bigs=[0,1,2], rows=[0,0,1], cols=[0,1,1]
  // Expected local (col-major with width=32):
  //   local[0*32+0] = 1.0  (R[0][0], lower tri)
  //   local[1*32+0] = 2.0  (R[0][1] lower: col=1,row=0)
  //   local[0*32+1] = 2.0  (R[0][1] upper: col=0,row=1)
  //   local[1*32+1] = 3.0  (R[1][1])
  {
    std::vector<double> global = {1.0, 2.0, 3.0};
    std::vector<uint> bigs = {0, 1, 2}, rows = {0, 0, 1}, cols = {0, 1, 1};
    auto got = run_gather(global, bigs, rows, cols, 2);
    uint w = COALESCED_DIMENSION(2);
    bool ok = fabsf(got[0*w+0] - 1.f) < tol &&
              fabsf(got[1*w+0] - 2.f) < tol &&
              fabsf(got[0*w+1] - 2.f) < tol &&   // mirror
              fabsf(got[1*w+1] - 3.f) < tol;
    runner.check(ok, "m=2 hand-verify lower+upper");
  }

  // --- 3. m=4: vs CPU reference ---
  {
    uint M = 4;
    uint packed_size = M * (M + 1) / 2;
    std::vector<double> global(packed_size);
    for (uint k = 0; k < packed_size; k++)
      global[k] = (double)(k + 1) * 0.7;

    std::vector<uint> bigs, rows, cols;
    build_full_indexes(M, bigs, rows, cols);

    auto got = run_gather(global, bigs, rows, cols, M);
    auto ref = ref_gather(global, bigs, rows, cols, M);

    bool ok = true;
    for (size_t i = 0; i < ref.size(); i++)
      if (fabsf(got[i] - ref[i]) > tol) { ok = false; break; }
    runner.check(ok, "m=4 vs CPU reference");
  }

  // --- 4. m=30: realistic group size, verify symmetry ---
  {
    uint M = 30;
    uint packed_size = M * (M + 1) / 2;
    std::vector<double> global(packed_size);
    for (uint k = 0; k < packed_size; k++)
      global[k] = sin((double)k * 0.1) * 0.5;

    std::vector<uint> bigs, rows, cols;
    build_full_indexes(M, bigs, rows, cols);

    auto got = run_gather(global, bigs, rows, cols, M);
    uint w = COALESCED_DIMENSION(M);

    // Check symmetry: local[i*w+j] == local[j*w+i] for all i,j < M
    bool symmetric = true;
    for (uint i = 0; i < M && symmetric; i++)
      for (uint j = 0; j < M && symmetric; j++)
        if (fabsf(got[i*w+j] - got[j*w+i]) > tol) symmetric = false;

    // Also check values match reference
    auto ref = ref_gather(global, bigs, rows, cols, M);
    bool values_ok = true;
    for (size_t i = 0; i < ref.size(); i++)
      if (fabsf(got[i] - ref[i]) > tol) { values_ok = false; break; }

    runner.check(symmetric && values_ok, "m=30 symmetry + values");
  }

  // --- 5. m=4: padding stays zero ---
  {
    uint M = 4;
    uint packed_size = M * (M + 1) / 2;
    std::vector<double> global(packed_size);
    for (uint k = 0; k < packed_size; k++)
      global[k] = 1.0 + (double)k;

    std::vector<uint> bigs, rows, cols;
    build_full_indexes(M, bigs, rows, cols);

    auto got = run_gather(global, bigs, rows, cols, M);
    uint w = COALESCED_DIMENSION(M);
    uint h = M + DENSITY_BLOCK_SIZE;

    // Check that elements outside the M×M block are zero
    bool padding_ok = true;
    for (uint col = 0; col < h; col++) {
      for (uint row = 0; row < w; row++) {
        if (row >= M || col >= M) {
          if (fabsf(got[col * w + row]) > tol) {
            padding_ok = false;
            break;
          }
        }
      }
      if (!padding_ok) break;
    }
    runner.check(padding_ok, "m=4 padding stays zero");
  }

  // --- 6. Subset gather: group uses only some global functions ---
  {
    uint M = 10;  // global M
    uint packed_size = M * (M + 1) / 2;
    std::vector<double> global(packed_size);
    for (uint k = 0; k < packed_size; k++)
      global[k] = (double)(k + 1) * 0.3;

    // Group uses global funcs {2, 5, 7} → local funcs {0, 1, 2}
    uint group_m = 3;
    uint global_funcs[] = {2, 5, 7};

    std::vector<uint> bigs, rows, cols;
    for (uint li = 0; li < group_m; li++) {
      for (uint lj = li; lj < group_m; lj++) {
        uint gi = global_funcs[li], gj = global_funcs[lj];
        // Ensure gi <= gj for packed index
        if (gi > gj) { uint t = gi; gi = gj; gj = t; }
        bigs.push_back(packed_index(gi, gj, M));
        rows.push_back(li);
        cols.push_back(lj);
      }
    }

    auto got = run_gather(global, bigs, rows, cols, group_m);
    auto ref = ref_gather(global, bigs, rows, cols, group_m);

    bool ok = true;
    for (size_t i = 0; i < ref.size(); i++)
      if (fabsf(got[i] - ref[i]) > tol) { ok = false; break; }
    runner.check(ok, "subset gather (3 of 10 global funcs)");
  }

  return runner.summary();
}
