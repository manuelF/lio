// Unit tests for g2g/cuda/kernels/rmm.h
//
// gpu_update_rmm computes the lower-triangle of a symmetric matrix:
//
//   RMM[j * COALESCED_DIMENSION(m) + i] = sum_p  factor[p] * F_i[p] * F_j[p]
//   for all i <= j < m
//
// function_values is laid out as F[func_idx * COALESCED_DIMENSION(points) + p].
//
// Two instantiations are tested:
//   check_pos=false  — standard rectangular grid, one thread per (i,j) pair.
//   check_pos=true   — triangular block indexing to skip the upper triangle.
//
// Test coverage
// -------------
//   1. Trivial: m=1, points=1
//   2. Hand-verifiable: m=2, points=3
//   3. Non-trivial: m=4, points=5 (float and double)
//   4. Full block: m=16 = RMM_BLOCK_SIZE_XY, points=32
//   5. Multi-outer-loop: m=4, points=300 (> RMM_BLOCK_SIZE_XY²)
//   6. Zero factors → zero RMM
//   7. Orthogonal functions → purely diagonal RMM
//   8. check_pos=true with m=2 (single triangular block)
//   9. check_pos=true with m=32 (three triangular blocks)

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC 0

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../../../g2g/common.h"  // RMM_BLOCK_SIZE_XY, DENSITY_*
#include "../../../g2g/matrix.h"  // COALESCED_DIMENSION, cuda_extra.h (index())
#include "test_utils.h"

namespace G2G {
#include "../../../g2g/cuda/kernels/rmm.h"
}

// ---------------------------------------------------------------------------
// CPU reference: RMM(i,j) = sum_p factor[p] * F_i[p] * F_j[p],  i <= j
// Outputs into rmm_ref[COALESCED_DIMENSION(m) * j + i].
// ---------------------------------------------------------------------------
template <typename T>
static void cpu_rmm(const T* factors, int points, const T* fv, int m,
                    T* rmm_ref) {
  int cdim_p = COALESCED_DIMENSION(points);
  int cdim_m = COALESCED_DIMENSION(m);
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i <= j; ++i) {
      T s = 0;
      for (int p = 0; p < points; ++p)
        s += factors[p] * fv[i * cdim_p + p] * fv[j * cdim_p + p];
      rmm_ref[j * cdim_m + i] = s;
    }
  }
}

// ---------------------------------------------------------------------------
// Run gpu_update_rmm<T, check_pos> and verify against the CPU reference.
// Returns true if all lower-triangle elements match within tol.
// ---------------------------------------------------------------------------
template <typename T, bool check_pos>
static bool run_rmm_test(int m, int points, T tol = T(1e-4)) {
  int cdim_p = COALESCED_DIMENSION(points);
  int cdim_m = COALESCED_DIMENSION(m);

  // Host buffers
  std::vector<T> h_factors(points);
  std::vector<T> h_fv(m * cdim_p, T(0));
  std::vector<T> h_rmm(cdim_m * m, T(0));
  std::vector<T> h_ref(cdim_m * m, T(0));

  // Fill factors and function values with non-trivial values
  for (int p = 0; p < points; ++p) h_factors[p] = T(p + 1);
  for (int fi = 0; fi < m; ++fi)
    for (int p = 0; p < points; ++p)
      h_fv[fi * cdim_p + p] = T((fi + 1) * (p + 1));

  cpu_rmm(h_factors.data(), points, h_fv.data(), m, h_ref.data());

  // Device allocations
  T *d_factors, *d_fv, *d_rmm;
  CUDA_CHECK(cudaMalloc(&d_factors, points * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_fv, m * cdim_p * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_rmm, cdim_m * m * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(d_factors, h_factors.data(), points * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fv, h_fv.data(), m * cdim_p * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_rmm, 0, cdim_m * m * sizeof(T)));

  // Launch
  dim3 block(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
  if (check_pos) {
    int n_tiles = (m + RMM_BLOCK_SIZE_XY - 1) / RMM_BLOCK_SIZE_XY;
    int n_blocks = n_tiles * (n_tiles + 1) / 2;
    dim3 grid(n_blocks, 1);
    G2G::gpu_update_rmm<T, true>
        <<<grid, block>>>(d_factors, points, d_rmm, d_fv, m);
  } else {
    int tiles = (m + RMM_BLOCK_SIZE_XY - 1) / RMM_BLOCK_SIZE_XY;
    dim3 grid(tiles, tiles);
    G2G::gpu_update_rmm<T, false>
        <<<grid, block>>>(d_factors, points, d_rmm, d_fv, m);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_rmm.data(), d_rmm, cdim_m * m * sizeof(T),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_factors);
  cudaFree(d_fv);
  cudaFree(d_rmm);

  // Verify lower triangle
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i <= j; ++i) {
      T got = h_rmm[j * cdim_m + i];
      T exp = h_ref[j * cdim_m + i];
      if (std::abs(got - exp) > tol * (T(1) + std::abs(exp))) {
        printf("    MISMATCH RMM(%d,%d): expected %.6g  got %.6g\n", i, j,
               (double)exp, (double)got);
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Special test: zero factors → every RMM element must be zero.
// ---------------------------------------------------------------------------
template <typename T>
static bool test_zero_factors(int m, int points) {
  int cdim_p = COALESCED_DIMENSION(points);
  int cdim_m = COALESCED_DIMENSION(m);

  std::vector<T> h_factors(points, T(0));
  std::vector<T> h_fv(m * cdim_p);
  std::vector<T> h_rmm(cdim_m * m, T(-1));  // pre-fill with sentinel

  for (int fi = 0; fi < m; ++fi)
    for (int p = 0; p < points; ++p)
      h_fv[fi * cdim_p + p] = T(fi * points + p + 1);

  T *d_factors, *d_fv, *d_rmm;
  CUDA_CHECK(cudaMalloc(&d_factors, points * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_fv, m * cdim_p * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_rmm, cdim_m * m * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_factors, h_factors.data(), points * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fv, h_fv.data(), m * cdim_p * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_rmm, 0, cdim_m * m * sizeof(T)));

  int tiles = (m + RMM_BLOCK_SIZE_XY - 1) / RMM_BLOCK_SIZE_XY;
  dim3 block(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
  dim3 grid(tiles, tiles);
  G2G::gpu_update_rmm<T, false>
      <<<grid, block>>>(d_factors, points, d_rmm, d_fv, m);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_rmm.data(), d_rmm, cdim_m * m * sizeof(T),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_factors);
  cudaFree(d_fv);
  cudaFree(d_rmm);

  for (int j = 0; j < m; ++j)
    for (int i = 0; i <= j; ++i)
      if (h_rmm[j * cdim_m + i] != T(0)) {
        printf("    Expected 0 at (%d,%d), got %.6g\n", i, j,
               (double)h_rmm[j * cdim_m + i]);
        return false;
      }
  return true;
}

// ---------------------------------------------------------------------------
// Special test: orthogonal functions → purely diagonal RMM.
// F_i[p] = 1 if p == i, else 0  (requires points >= m).
// factor[p] = 1 for all p.
// Expected RMM: diagonal = 1, off-diagonal = 0.
// ---------------------------------------------------------------------------
template <typename T>
static bool test_orthogonal_functions(int m) {
  int points = m;  // one-to-one: function i is non-zero only at point i
  int cdim_p = COALESCED_DIMENSION(points);
  int cdim_m = COALESCED_DIMENSION(m);

  std::vector<T> h_factors(points, T(1));
  std::vector<T> h_fv(m * cdim_p, T(0));
  std::vector<T> h_rmm(cdim_m * m, T(0));

  for (int fi = 0; fi < m; ++fi)
    h_fv[fi * cdim_p + fi] = T(1);  // F_i[i] = 1, all others 0

  T *d_factors, *d_fv, *d_rmm;
  CUDA_CHECK(cudaMalloc(&d_factors, points * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_fv, m * cdim_p * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_rmm, cdim_m * m * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_factors, h_factors.data(), points * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fv, h_fv.data(), m * cdim_p * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_rmm, 0, cdim_m * m * sizeof(T)));

  int tiles = (m + RMM_BLOCK_SIZE_XY - 1) / RMM_BLOCK_SIZE_XY;
  dim3 block(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
  dim3 grid(tiles, tiles);
  G2G::gpu_update_rmm<T, false>
      <<<grid, block>>>(d_factors, points, d_rmm, d_fv, m);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_rmm.data(), d_rmm, cdim_m * m * sizeof(T),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_factors);
  cudaFree(d_fv);
  cudaFree(d_rmm);

  for (int j = 0; j < m; ++j) {
    for (int i = 0; i <= j; ++i) {
      T got = h_rmm[j * cdim_m + i];
      T exp = (i == j) ? T(1) : T(0);
      if (got != exp) {
        printf("    Orthogonal: expected RMM(%d,%d)=%.0f, got %.6g\n", i, j,
               (double)exp, (double)got);
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Precision test: compare GPU float against CPU double reference.
// Uses positive monotonic data (like production: all-positive function values
// and factors) to avoid cancellation that inflates relative error.
// Prints measured max relative error for diagnostics.
// ---------------------------------------------------------------------------
template <bool check_pos>
static bool test_precision(int m, int points, float tol, const char* label) {
  int cdim_p = COALESCED_DIMENSION(points);
  int cdim_m = COALESCED_DIMENSION(m);

  // Host buffers — positive, varied, O(1) magnitude (realistic for DFT)
  std::vector<float> h_factors(points);
  std::vector<float> h_fv(m * cdim_p, 0.0f);
  std::vector<float> h_rmm(cdim_m * m, 0.0f);

  for (int p = 0; p < points; ++p)
    h_factors[p] = 0.01f + 0.1f * (p % 7);  // O(0.01–0.7), always positive
  for (int fi = 0; fi < m; ++fi)
    for (int p = 0; p < points; ++p)
      h_fv[fi * cdim_p + p] = 0.1f + 0.05f * ((fi + p) % 11);  // O(0.1–0.6)

  // Double-precision CPU reference (using exact same float inputs)
  std::vector<double> d_factors(points);
  std::vector<double> d_fv(m * cdim_p, 0.0);
  std::vector<double> d_ref(cdim_m * m, 0.0);
  for (int p = 0; p < points; ++p) d_factors[p] = (double)h_factors[p];
  for (int fi = 0; fi < m; ++fi)
    for (int p = 0; p < points; ++p)
      d_fv[fi * cdim_p + p] = (double)h_fv[fi * cdim_p + p];
  cpu_rmm(d_factors.data(), points, d_fv.data(), m, d_ref.data());

  // GPU computation in float
  float *df, *dfv, *drmm;
  CUDA_CHECK(cudaMalloc(&df, points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dfv, m * cdim_p * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&drmm, cdim_m * m * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(df, h_factors.data(), points * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dfv, h_fv.data(), m * cdim_p * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(drmm, 0, cdim_m * m * sizeof(float)));

  dim3 block(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
  if (check_pos) {
    int n_tiles = (m + RMM_BLOCK_SIZE_XY - 1) / RMM_BLOCK_SIZE_XY;
    int n_blocks = n_tiles * (n_tiles + 1) / 2;
    G2G::gpu_update_rmm<float, true>
        <<<dim3(n_blocks, 1), block>>>(df, points, drmm, dfv, m);
  } else {
    int tiles = (m + RMM_BLOCK_SIZE_XY - 1) / RMM_BLOCK_SIZE_XY;
    G2G::gpu_update_rmm<float, false>
        <<<dim3(tiles, tiles), block>>>(df, points, drmm, dfv, m);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_rmm.data(), drmm, cdim_m * m * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(df);
  cudaFree(dfv);
  cudaFree(drmm);

  // Measure max relative error
  double max_rel_err = 0.0;
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i <= j; ++i) {
      double got = (double)h_rmm[j * cdim_m + i];
      double ref = d_ref[j * cdim_m + i];
      double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
      double rel = fabs(got - ref) / denom;
      if (rel > max_rel_err) max_rel_err = rel;
    }
  }
  printf("    %s: max_rel_err = %.2e (tol %.1e)\n", label, max_rel_err, (double)tol);
  return max_rel_err < tol;
}

// ---------------------------------------------------------------------------
// Bit-exact reproducibility: dump FNV-1a hash of the raw GPU output bytes
// for a representative kernel run. A kernel modification that is truly
// FP-neutral (e.g. an algebraic no-op refactor) must produce the identical
// hash. Print the hash so it can be compared across code versions.
// ---------------------------------------------------------------------------
template <bool check_pos>
static void dump_bit_hash(int m, int points, const char* label) {
  int cdim_p = COALESCED_DIMENSION(points);
  int cdim_m = COALESCED_DIMENSION(m);

  std::vector<float> h_factors(points);
  std::vector<float> h_fv(m * cdim_p, 0.0f);
  std::vector<float> h_rmm(cdim_m * m, 0.0f);

  for (int p = 0; p < points; ++p)
    h_factors[p] = 0.01f + 0.1f * (p % 7);
  for (int fi = 0; fi < m; ++fi)
    for (int p = 0; p < points; ++p)
      h_fv[fi * cdim_p + p] = 0.1f + 0.05f * ((fi + p) % 11);

  float *df, *dfv, *drmm;
  CUDA_CHECK(cudaMalloc(&df, points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dfv, m * cdim_p * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&drmm, cdim_m * m * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(df, h_factors.data(), points * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dfv, h_fv.data(), m * cdim_p * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(drmm, 0, cdim_m * m * sizeof(float)));

  dim3 block(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
  if (check_pos) {
    int n_tiles = (m + RMM_BLOCK_SIZE_XY - 1) / RMM_BLOCK_SIZE_XY;
    int n_blocks = n_tiles * (n_tiles + 1) / 2;
    G2G::gpu_update_rmm<float, true>
        <<<dim3(n_blocks, 1), block>>>(df, points, drmm, dfv, m);
  } else {
    int tiles = (m + RMM_BLOCK_SIZE_XY - 1) / RMM_BLOCK_SIZE_XY;
    G2G::gpu_update_rmm<float, false>
        <<<dim3(tiles, tiles), block>>>(df, points, drmm, dfv, m);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_rmm.data(), drmm, cdim_m * m * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(df);
  cudaFree(dfv);
  cudaFree(drmm);

  // Hash only the lower-triangle cells (the kernel may leave upper undefined)
  uint64_t h = 1469598103934665603ULL;  // FNV offset
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i <= j; ++i) {
      uint32_t bits;
      std::memcpy(&bits, &h_rmm[j * cdim_m + i], 4);
      for (int b = 0; b < 4; ++b) {
        h ^= (bits >> (8 * b)) & 0xFF;
        h *= 1099511628211ULL;
      }
    }
  }
  printf("    bit-hash %s  m=%d pts=%d  0x%016lx\n", label, m, points,
         (unsigned long)h);
}

int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_update_rmm kernel");

  // --- check_pos=false ---
  printf("[ check_pos=false ]\n");
  runner.check(run_rmm_test<float, false>(1, 1), "float  m=1  pts=1   trivial");
  runner.check(run_rmm_test<float, false>(2, 3),
               "float  m=2  pts=3   hand-verifiable");
  runner.check(run_rmm_test<float, false>(4, 5), "float  m=4  pts=5");
  runner.check(run_rmm_test<float, false>(4, 300),
               "float  m=4  pts=300 multi-outer-loop");
  runner.check(run_rmm_test<float, false>(16, 32),
               "float  m=16 pts=32  full block");
  runner.check(run_rmm_test<float, false>(16, 300),
               "float  m=16 pts=300 full block, multi-outer-loop");
  runner.check(run_rmm_test<double, false>(2, 3), "double m=2  pts=3");
  runner.check(run_rmm_test<double, false>(4, 5), "double m=4  pts=5");
  runner.check(run_rmm_test<double, false>(16, 32),
               "double m=16 pts=32  full block");

  printf("\n[ special cases ]\n");
  runner.check(test_zero_factors<float>(4, 5),
               "float  zero factors  → zero RMM");
  runner.check(test_zero_factors<double>(4, 5),
               "double zero factors  → zero RMM");
  runner.check(test_orthogonal_functions<float>(4),
               "float  orthogonal Fs → diagonal RMM");
  runner.check(test_orthogonal_functions<float>(16),
               "float  orthogonal Fs → diagonal RMM (full block)");

  // --- check_pos=true (triangular block indexing) ---
  printf("\n[ check_pos=true ]\n");
  runner.check(run_rmm_test<float, true>(2, 3),
               "float  m=2  pts=3   single tri-block");
  runner.check(run_rmm_test<float, true>(16, 32),
               "float  m=16 pts=32  single tri-block (full)");
  runner.check(run_rmm_test<float, true>(32, 50),
               "float  m=32 pts=50  three tri-blocks");
  runner.check(run_rmm_test<double, true>(2, 3),
               "double m=2  pts=3   single tri-block");
  runner.check(run_rmm_test<double, true>(32, 50),
               "double m=32 pts=50  three tri-blocks");

  // --- precision tests (float GPU vs double CPU) ---
  printf("\n[ precision: float GPU vs double CPU ]\n");
  runner.check(test_precision<false>(16, 500, 1e-5f, "m=16 pts=500"),
               "float precision m=16 pts=500 (rel_err < 1e-5)");
  runner.check(test_precision<true>(16, 500, 1e-5f, "m=16 pts=500 check_pos"),
               "float precision m=16 pts=500 check_pos (rel_err < 1e-5)");

  printf("\n[ bit-exact reproducibility hashes ]\n");
  dump_bit_hash<false>(16, 500, "check_pos=false");
  dump_bit_hash<true>(16, 500, "check_pos=true ");
  dump_bit_hash<false>(33, 97, "check_pos=false");  // non-multiple-of-16
  dump_bit_hash<true>(33, 97, "check_pos=true ");
  dump_bit_hash<false>(64, 500, "check_pos=false");
  dump_bit_hash<true>(64, 500, "check_pos=true ");
  dump_bit_hash<false>(128, 1000, "check_pos=false");
  dump_bit_hash<true>(128, 1000, "check_pos=true ");

  return runner.summary();
}
