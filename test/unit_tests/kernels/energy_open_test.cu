// Unit tests for g2g/cuda/kernels/energy_open.h
//
// gpu_compute_density_opened<scalar_type, compute_energy, compute_factor, lda>
// computes alpha and beta densities for open-shell systems from two separate
// RMM textures using the same basis-function values:
//
//   rho_a(p) = sum_i F_i(p) * sum_{j<=i} Ra[i][j] * F_j(p)
//   rho_b(p) = sum_i F_i(p) * sum_{j<=i} Rb[i][j] * F_j(p)
//
// function_values layout: fv[m * point + func_idx]  (same as energy.h)
// Two RMM textures: tex_a and tex_b, both with fetch(t, col=j, row=i)=R[i][j]
// Output per block row: out_partial_density_{a,b}[blockIdx.y * points + blockIdx.x]
//
// Block = dim3(DENSITY_BLOCK_SIZE=64), Grid = dim3(points, n_block_rows)
// n_block_rows = ceil(m / (2*DENSITY_BLOCK_SIZE))
//
// Test coverage
// -------------
//   1. m=1, pts=1, lda=true:  trivial  rho_a, rho_b
//   2. m=2, pts=1, lda=true:  hand-verifiable alpha and beta
//   3. m=4, pts=3, lda=true:  vs CPU reference (alpha and beta)
//   4. Ra==Rb: same function values → rho_a == rho_b
//   5. m=130, pts=2, lda=true: two block rows vs CPU

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC   0

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "test_utils.h"
#include "../../../g2g/common.h"               // DENSITY_BLOCK_SIZE
#include "../../../g2g/matrix.h"               // COALESCED_DIMENSION
#include "../../../g2g/scalar_vector_types.h"  // vec_type<T,N>
#include "../../../g2g/cuda/cuda_extra.h"      // index_x

// energy.h defines warpReduceScalar, warpReduceVector3, and fetch.
// energy_open.h redefines fetch (same value for FULL_DOUBLE=0) and provides
// gpu_compute_density_opened.
namespace G2G {
#include "../../../g2g/cuda/kernels/energy.h"
#include "../../../g2g/cuda/kernels/energy_open.h"
}

using F4 = G2G::vec_type<float, 4>;

// ---------------------------------------------------------------------------
// Create a CUDA 2D texture from a row-major float[m*m] array.
// Caller must destroy texObj and free cuArray.
// ---------------------------------------------------------------------------
static cudaTextureObject_t make_rmm_texture(const std::vector<float>& rmm,
                                            int m, cudaArray_t& cuArray) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaMallocArray(&cuArray, &desc, m, m));
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, rmm.data(), m * sizeof(float),
                                 m * sizeof(float), m, cudaMemcpyHostToDevice));
  cudaResourceDesc resDesc{};
  resDesc.resType         = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;
  cudaTextureDesc texDesc{};
  texDesc.readMode        = cudaReadModeElementType;
  texDesc.filterMode      = cudaFilterModePoint;
  texDesc.addressMode[0]  = cudaAddressModeClamp;
  texDesc.addressMode[1]  = cudaAddressModeClamp;
  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
  return texObj;
}

// ---------------------------------------------------------------------------
// CPU reference: closed-shell density (used for both alpha and beta slices).
// rho(p) = sum_i fv[m*p+i] * sum_{j<=i} rmm[i*m+j] * fv[m*p+j]
// ---------------------------------------------------------------------------
static float cpu_density(const std::vector<float>& rmm, int m,
                         const std::vector<float>& fv,  int p) {
  float rho = 0.f;
  for (int i = 0; i < m; ++i) {
    float wi = 0.f;
    for (int j = 0; j <= i; ++j) wi += rmm[i * m + j] * fv[m * p + j];
    rho += fv[m * p + i] * wi;
  }
  return rho;
}

// ---------------------------------------------------------------------------
// Run gpu_compute_density_opened<float, false, false, lda=true>.
// Returns {rho_a_per_point, rho_b_per_point}.
// ---------------------------------------------------------------------------
static std::pair<std::vector<float>, std::vector<float>>
run_density_open(const std::vector<float>& rmm_a, const std::vector<float>& rmm_b,
                 int m, const std::vector<float>& fv, int pts) {
  int n_rows   = (m + 2 * DENSITY_BLOCK_SIZE - 1) / (2 * DENSITY_BLOCK_SIZE);
  int out_size = n_rows * pts;

  float* d_fv;
  CUDA_CHECK(cudaMalloc(&d_fv, (size_t)m * pts * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_fv, fv.data(), (size_t)m * pts * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaArray_t ca_a, ca_b;
  cudaTextureObject_t tex_a = make_rmm_texture(rmm_a, m, ca_a);
  cudaTextureObject_t tex_b = make_rmm_texture(rmm_b, m, ca_b);

  float *d_pd_a, *d_pd_b;
  F4    *d_dxyz_a, *d_dd1_a, *d_dd2_a;
  F4    *d_dxyz_b, *d_dd1_b, *d_dd2_b;
  CUDA_CHECK(cudaMalloc(&d_pd_a,   out_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pd_b,   out_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dxyz_a, out_size * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dd1_a,  out_size * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dd2_a,  out_size * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dxyz_b, out_size * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dd1_b,  out_size * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dd2_b,  out_size * sizeof(F4)));

  dim3 block(DENSITY_BLOCK_SIZE);
  dim3 grid(pts, n_rows);
  // compute_energy=false, compute_factor=false, lda=true
  G2G::gpu_compute_density_opened<float, false, false, true><<<grid, block>>>(
      tex_a, tex_b, nullptr, pts,
      d_fv, nullptr, nullptr, m,
      d_pd_a, d_dxyz_a, d_dd1_a, d_dd2_a,
      d_pd_b, d_dxyz_b, d_dd1_b, d_dd2_b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_a(out_size), h_b(out_size);
  CUDA_CHECK(cudaMemcpy(h_a.data(), d_pd_a, out_size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_b.data(), d_pd_b, out_size * sizeof(float), cudaMemcpyDeviceToHost));

  cudaDestroyTextureObject(tex_a); cudaFreeArray(ca_a);
  cudaDestroyTextureObject(tex_b); cudaFreeArray(ca_b);
  cudaFree(d_fv);
  cudaFree(d_pd_a); cudaFree(d_dxyz_a); cudaFree(d_dd1_a); cudaFree(d_dd2_a);
  cudaFree(d_pd_b); cudaFree(d_dxyz_b); cudaFree(d_dd1_b); cudaFree(d_dd2_b);

  // Sum block rows
  std::vector<float> rho_a(pts, 0.f), rho_b(pts, 0.f);
  for (int r = 0; r < n_rows; ++r)
    for (int p = 0; p < pts; ++p) {
      rho_a[p] += h_a[r * pts + p];
      rho_b[p] += h_b[r * pts + p];
    }
  return {rho_a, rho_b};
}

// ============================================================================
int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_compute_density_opened kernel");

  const float tol = 1e-4f;

  // --- 1. m=1, pts=1: rho_a=Ra[0][0]*F[0]^2=6, rho_b=Rb[0][0]*F[0]^2=12 ---
  {
    auto p1 = run_density_open({2.f}, {4.f}, 1, {3.f}, 1);
    runner.check(fabsf(p1.first[0] - 18.f) < tol && fabsf(p1.second[0] - 36.f) < tol,
                 "m=1 pts=1 trivial rho_a=18 rho_b=36");
  }

  // --- 2. m=2, pts=1: hand-verify ---
  // Ra: R[0][0]=1, R[1][0]=2, R[1][1]=3    Rb: R[0][0]=2, R[1][0]=0, R[1][1]=1
  // F=[1,2]
  // rho_a = 1*1 + 2*(2+6) = 17    rho_b = 1*2 + 2*(0+2) = 6
  {
    std::vector<float> ra = {1.f, 0.f, 2.f, 3.f};
    std::vector<float> rb = {2.f, 0.f, 0.f, 1.f};
    auto p2 = run_density_open(ra, rb, 2, {1.f, 2.f}, 1);
    runner.check(fabsf(p2.first[0] - 17.f) < tol && fabsf(p2.second[0] - 6.f) < tol,
                 "m=2 pts=1 hand-verify rho_a=17 rho_b=6");
  }

  // --- 3. m=4, pts=3: vs CPU ---
  {
    int m = 4, pts = 3;
    std::vector<float> ra(m * m, 0.f), rb(m * m, 0.f), fv(m * pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j) {
        ra[i * m + j] = float(i * m + j + 1) * 0.1f;
        rb[i * m + j] = float(i * m + j + 2) * 0.07f;
      }
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m * p + i] = float(p * m + i + 1) * 0.25f;
    auto p3 = run_density_open(ra, rb, m, fv, pts);
    bool ok = true;
    for (int p = 0; p < pts; ++p) {
      ok &= fabsf(p3.first[p]  - cpu_density(ra, m, fv, p)) < tol;
      ok &= fabsf(p3.second[p] - cpu_density(rb, m, fv, p)) < tol;
    }
    runner.check(ok, "m=4 pts=3 vs CPU");
  }

  // --- 4. Ra==Rb → rho_a == rho_b ---
  {
    int m = 8, pts = 4;
    std::vector<float> r(m * m, 0.f), fv(m * pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        r[i * m + j] = sinf(float(i + j) * 0.3f);
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m * p + i] = cosf(float(p * m + i) * 0.1f);
    auto p4 = run_density_open(r, r, m, fv, pts);
    bool ok = true;
    for (int p = 0; p < pts; ++p) ok &= fabsf(p4.first[p] - p4.second[p]) < tol;
    runner.check(ok, "Ra==Rb → rho_a == rho_b");
  }

  // --- 5. m=130, pts=2: two block rows vs CPU ---
  {
    int m = 130, pts = 2;
    std::vector<float> ra(m * m, 0.f), rb(m * m, 0.f), fv(m * pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j) {
        ra[i * m + j] = float(i + j + 1) * 0.004f;
        rb[i * m + j] = float(i + j + 2) * 0.003f;
      }
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m * p + i] = float(p + i + 1) * 0.008f;
    auto p5 = run_density_open(ra, rb, m, fv, pts);
    bool ok = true;
    for (int p = 0; p < pts; ++p) {
      ok &= fabsf(p5.first[p]  - cpu_density(ra, m, fv, p)) < 5e-2f;
      ok &= fabsf(p5.second[p] - cpu_density(rb, m, fv, p)) < 5e-2f;
    }
    runner.check(ok, "m=130 pts=2 two block rows vs CPU");
  }

  return runner.summary();
}
