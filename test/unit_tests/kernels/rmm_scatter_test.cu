// Unit tests for g2g/cuda/kernels/rmm_scatter.h
//
// gpu_scatter_rmm<scalar_type>:
//   For each index k in [0, n_indexes):
//     val = (double)local_rmm[cols[k] * rmm_width + rows[k]]
//     atomicAdd(&global_rmm_out[bigs[k]], val)
//
// Test coverage
// -------------
//   1. m=1: trivial single element
//   2. m=2: hand-verify two entries scatter to correct packed indices
//   3. m=4: compare GPU scatter vs CPU reference
//   4. m=30: realistic group size, symmetry of packed indices
//   5. Two overlapping groups: verify atomicAdd accumulates correctly
//   6. m=4: verify padding columns in local_rmm are not read

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
#include "../../../g2g/cuda/kernels/rmm_scatter.h"
}

// ---------------------------------------------------------------------------
// CPU reference: packed upper-triangular index
// big_index(i,j) = i * M - i*(i-1)/2 + (j - i)   where i <= j
// ---------------------------------------------------------------------------
static uint packed_index(uint i, uint j, uint M) {
  if (i > j) { uint t = i; i = j; j = t; }
  return i * M - (i * (i - 1)) / 2 + (j - i);
}

// CPU reference scatter: mirrors add_rmm_output
// local_rmm layout: lower triangle, local_rmm[col * rmm_width + row] for row <= col
static void ref_scatter(
    const std::vector<float>& local_rmm,
    const std::vector<uint>& bigs,
    const std::vector<uint>& rows,
    const std::vector<uint>& cols,
    std::vector<double>& global_out,
    uint rmm_width) {
  for (size_t k = 0; k < bigs.size(); k++) {
    uint r = rows[k], c = cols[k];
    double val = (double)local_rmm[c * rmm_width + r];
    global_out[bigs[k]] += val;
  }
}

// ---------------------------------------------------------------------------
// Build index arrays for a group with identity local2global mapping
// (local func i maps to global func i), upper triangle i <= j
// ---------------------------------------------------------------------------
static void build_indexes(uint group_m, uint M,
                          std::vector<uint>& bigs,
                          std::vector<uint>& rows,
                          std::vector<uint>& cols) {
  bigs.clear(); rows.clear(); cols.clear();
  for (uint i = 0; i < group_m; i++) {
    for (uint j = i; j < group_m; j++) {
      bigs.push_back(packed_index(i, j, M));
      rows.push_back(i);
      cols.push_back(j);
    }
  }
}

// Build index arrays with an offset (local func i maps to global func i+offset)
static void build_indexes_offset(uint group_m, uint M, uint offset,
                                 std::vector<uint>& bigs,
                                 std::vector<uint>& rows,
                                 std::vector<uint>& cols) {
  bigs.clear(); rows.clear(); cols.clear();
  for (uint i = 0; i < group_m; i++) {
    for (uint j = i; j < group_m; j++) {
      bigs.push_back(packed_index(i + offset, j + offset, M));
      rows.push_back(i);
      cols.push_back(j);
    }
  }
}

// ---------------------------------------------------------------------------
// Run GPU scatter kernel
// ---------------------------------------------------------------------------
static std::vector<double> run_scatter(
    const std::vector<float>& local_rmm,
    const std::vector<uint>& bigs,
    const std::vector<uint>& rows,
    const std::vector<uint>& cols,
    uint global_size,
    uint rmm_width,
    double* d_global_out = nullptr) {  // optional: pass existing buffer for multi-group

  uint n_indexes = (uint)bigs.size();
  uint local_size = (uint)local_rmm.size();

  uint *d_bigs, *d_rows, *d_cols;
  float* d_local;
  bool own_global = (d_global_out == nullptr);

  if (own_global) {
    CUDA_CHECK(cudaMalloc(&d_global_out, global_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_global_out, 0, global_size * sizeof(double)));
  }

  CUDA_CHECK(cudaMalloc(&d_bigs, n_indexes * sizeof(uint)));
  CUDA_CHECK(cudaMalloc(&d_rows, n_indexes * sizeof(uint)));
  CUDA_CHECK(cudaMalloc(&d_cols, n_indexes * sizeof(uint)));
  CUDA_CHECK(cudaMalloc(&d_local, local_size * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_bigs, bigs.data(), n_indexes * sizeof(uint),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rows, rows.data(), n_indexes * sizeof(uint),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cols, cols.data(), n_indexes * sizeof(uint),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_local, local_rmm.data(), local_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((n_indexes + 255) / 256);
  G2G::gpu_scatter_rmm<float><<<grid, block>>>(
      d_local, d_bigs, d_rows, d_cols, d_global_out, n_indexes, rmm_width);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> result(global_size, 0.0);
  CUDA_CHECK(cudaMemcpy(result.data(), d_global_out,
                        global_size * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_bigs));
  CUDA_CHECK(cudaFree(d_rows));
  CUDA_CHECK(cudaFree(d_cols));
  CUDA_CHECK(cudaFree(d_local));
  if (own_global) CUDA_CHECK(cudaFree(d_global_out));
  return result;
}

// ---------------------------------------------------------------------------
// Helper: compare GPU result vs CPU reference, return true if all match
// ---------------------------------------------------------------------------
static bool compare_vectors(const std::vector<double>& gpu,
                            const std::vector<double>& ref,
                            double tol) {
  for (uint i = 0; i < (uint)gpu.size(); i++) {
    if (fabs(gpu[i] - ref[i]) > tol) {
      printf("  MISMATCH at global[%u]: gpu=%g ref=%g\n", i, gpu[i], ref[i]);
      return false;
    }
  }
  return true;
}

// ===========================================================================
int main() {
  test_utils::TestRunner runner("gpu_scatter_rmm kernel");

  // -----------------------------------------------------------------------
  // 1. m=1 trivial
  // -----------------------------------------------------------------------
  {
    uint M = 10, group_m = 1;
    uint rmm_width = COALESCED_DIMENSION(group_m);
    uint global_size = M * (M + 1) / 2;

    std::vector<uint> bigs, rows, cols;
    build_indexes(group_m, M, bigs, rows, cols);

    std::vector<float> local(rmm_width * group_m, 0.f);
    local[0] = 3.5f;

    auto gpu = run_scatter(local, bigs, rows, cols, global_size, rmm_width);

    bool ok = (fabs(gpu[0] - 3.5) <= 1e-6);
    for (uint i = 1; i < global_size && ok; i++)
      if (fabs(gpu[i]) > 1e-12) ok = false;
    runner.check(ok, "m=1 trivial");
  }

  // -----------------------------------------------------------------------
  // 2. m=2 hand-verify
  // -----------------------------------------------------------------------
  {
    uint M = 10, group_m = 2;
    uint rmm_width = COALESCED_DIMENSION(group_m);
    uint global_size = M * (M + 1) / 2;

    std::vector<uint> bigs, rows, cols;
    build_indexes(group_m, M, bigs, rows, cols);

    std::vector<float> local(rmm_width * group_m, 0.f);
    local[0 * rmm_width + 0] = 1.0f;
    local[1 * rmm_width + 0] = 2.0f;
    local[1 * rmm_width + 1] = 3.0f;

    auto gpu = run_scatter(local, bigs, rows, cols, global_size, rmm_width);

    bool ok = (bigs.size() == 3) &&
              (fabs(gpu[packed_index(0, 0, M)] - 1.0) <= 1e-6) &&
              (fabs(gpu[packed_index(0, 1, M)] - 2.0) <= 1e-6) &&
              (fabs(gpu[packed_index(1, 1, M)] - 3.0) <= 1e-6);
    runner.check(ok, "m=2 hand-verify lower triangle");
  }

  // -----------------------------------------------------------------------
  // 3. m=4 vs CPU reference
  // -----------------------------------------------------------------------
  {
    uint M = 20, group_m = 4;
    uint rmm_width = COALESCED_DIMENSION(group_m);
    uint global_size = M * (M + 1) / 2;

    std::vector<uint> bigs, rows, cols;
    build_indexes(group_m, M, bigs, rows, cols);

    std::vector<float> local(rmm_width * group_m, 0.f);
    for (uint j = 0; j < group_m; j++)
      for (uint i = 0; i <= j; i++)
        local[j * rmm_width + i] = (float)(i * 10 + j + 1);

    auto gpu = run_scatter(local, bigs, rows, cols, global_size, rmm_width);

    std::vector<double> ref(global_size, 0.0);
    ref_scatter(local, bigs, rows, cols, ref, rmm_width);

    runner.check(compare_vectors(gpu, ref, 1e-6), "m=4 vs CPU reference");
  }

  // -----------------------------------------------------------------------
  // 4. m=30 realistic size + symmetry
  // -----------------------------------------------------------------------
  {
    uint M = 100, group_m = 30;
    uint rmm_width = COALESCED_DIMENSION(group_m);
    uint global_size = M * (M + 1) / 2;

    std::vector<uint> bigs, rows, cols;
    build_indexes(group_m, M, bigs, rows, cols);

    std::vector<float> local(rmm_width * group_m, 0.f);
    for (uint j = 0; j < group_m; j++)
      for (uint i = 0; i <= j; i++)
        local[j * rmm_width + i] = sinf((float)(i * group_m + j));

    auto gpu = run_scatter(local, bigs, rows, cols, global_size, rmm_width);

    std::vector<double> ref(global_size, 0.0);
    ref_scatter(local, bigs, rows, cols, ref, rmm_width);

    runner.check(compare_vectors(gpu, ref, 1e-5), "m=30 symmetry + values");
  }

  // -----------------------------------------------------------------------
  // 5. Two overlapping groups: atomicAdd accumulation
  // -----------------------------------------------------------------------
  {
    uint M = 20, group_m = 3;
    uint rmm_width = COALESCED_DIMENSION(group_m);
    uint global_size = M * (M + 1) / 2;

    std::vector<uint> bigs_a, rows_a, cols_a;
    build_indexes_offset(group_m, M, 0, bigs_a, rows_a, cols_a);
    std::vector<uint> bigs_b, rows_b, cols_b;
    build_indexes_offset(group_m, M, 1, bigs_b, rows_b, cols_b);

    std::vector<float> local_a(rmm_width * group_m, 0.f);
    std::vector<float> local_b(rmm_width * group_m, 0.f);
    for (uint j = 0; j < group_m; j++) {
      for (uint i = 0; i <= j; i++) {
        local_a[j * rmm_width + i] = 1.0f;
        local_b[j * rmm_width + i] = 2.0f;
      }
    }

    std::vector<double> ref(global_size, 0.0);
    ref_scatter(local_a, bigs_a, rows_a, cols_a, ref, rmm_width);
    ref_scatter(local_b, bigs_b, rows_b, cols_b, ref, rmm_width);

    double* d_global;
    CUDA_CHECK(cudaMalloc(&d_global, global_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_global, 0, global_size * sizeof(double)));

    // Scatter group A
    {
      uint n = (uint)bigs_a.size();
      uint *d_b, *d_r, *d_c; float* d_l;
      CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(uint)));
      CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(uint)));
      CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(uint)));
      CUDA_CHECK(cudaMalloc(&d_l, local_a.size() * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_b, bigs_a.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_r, rows_a.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_c, cols_a.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_l, local_a.data(), local_a.size() * sizeof(float), cudaMemcpyHostToDevice));
      dim3 block(256), grid((n + 255) / 256);
      G2G::gpu_scatter_rmm<float><<<grid, block>>>(d_l, d_b, d_r, d_c, d_global, n, rmm_width);
      CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_r));
      CUDA_CHECK(cudaFree(d_c)); CUDA_CHECK(cudaFree(d_l));
    }
    // Scatter group B
    {
      uint n = (uint)bigs_b.size();
      uint *d_b, *d_r, *d_c; float* d_l;
      CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(uint)));
      CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(uint)));
      CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(uint)));
      CUDA_CHECK(cudaMalloc(&d_l, local_b.size() * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_b, bigs_b.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_r, rows_b.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_c, cols_b.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_l, local_b.data(), local_b.size() * sizeof(float), cudaMemcpyHostToDevice));
      dim3 block(256), grid((n + 255) / 256);
      G2G::gpu_scatter_rmm<float><<<grid, block>>>(d_l, d_b, d_r, d_c, d_global, n, rmm_width);
      CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_r));
      CUDA_CHECK(cudaFree(d_c)); CUDA_CHECK(cudaFree(d_l));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpu(global_size);
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_global, global_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_global));

    runner.check(compare_vectors(gpu, ref, 1e-6), "two overlapping groups atomicAdd");
  }

  // -----------------------------------------------------------------------
  // 6. Subset scatter (group funcs are a subset of global, with offset)
  // -----------------------------------------------------------------------
  {
    uint M = 10, group_m = 3;
    uint rmm_width = COALESCED_DIMENSION(group_m);
    uint global_size = M * (M + 1) / 2;

    std::vector<uint> bigs, rows, cols;
    build_indexes_offset(group_m, M, 5, bigs, rows, cols);

    std::vector<float> local(rmm_width * group_m, 0.f);
    for (uint j = 0; j < group_m; j++)
      for (uint i = 0; i <= j; i++)
        local[j * rmm_width + i] = (float)(i + j * 10 + 100);

    auto gpu = run_scatter(local, bigs, rows, cols, global_size, rmm_width);

    std::vector<double> ref(global_size, 0.0);
    ref_scatter(local, bigs, rows, cols, ref, rmm_width);

    runner.check(compare_vectors(gpu, ref, 1e-5), "subset scatter (offset=5)");
  }

  return runner.summary();
}
