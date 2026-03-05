// Unit tests for g2g/cuda/kernels/energy.h
//
// gpu_compute_density<scalar_type, lda> computes:
//
//   rho(p) = sum_i F_i(p) * sum_{j<=i} R[i][j] * F_j(p)
//
// function_values layout: fv[m * point + func_idx]
// RMM accessed via 2D texture: fetch(t, col=j, row=i) = R[i][j]
// Output (per block row): out_partial_density[blockIdx.y * points + blockIdx.x]
// Final density at point p = sum of all block-row contributions.
//
// Block = dim3(DENSITY_BLOCK_SIZE=64), Grid = dim3(points, n_block_rows)
// where n_block_rows = ceil(m / (2*DENSITY_BLOCK_SIZE)).
//
// Kahan compensated summation in the bj-loop reduces accumulation error
// from O(N*eps) to O(eps).
//
// For lda=false the kernel also produces out_dxyz/dd1/dd2 (gradient/Hessian
// contributions). energy, factor, point_weights params are unused by the kernel
// and can be nullptr.
//
// Test coverage
// -------------
//   1. m=1,   pts=1  lda=true:  trivial  rho = R[0][0]*F[0]^2
//   2. m=2,   pts=1  lda=true:  hand-verifiable
//   3. m=4,   pts=3  lda=true:  vs CPU reference
//   4. m=100, pts=2  lda=true:  two block rows (100 > DENSITY_BLOCK_SIZE)
//   5. m=130, pts=3  lda=true:  three block rows (130 > 2*DENSITY_BLOCK_SIZE)
//   6. m=2,   pts=1  lda=false: verify dxyz and dd1 outputs analytically
//   7. m=300, pts=1  lda=true:  FP precision test — GPU vs double CPU ref

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC   0

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "test_utils.h"
#include "kernels_reference.h"
#include "../../../g2g/common.h"               // DENSITY_BLOCK_SIZE
#include "../../../g2g/matrix.h"               // COALESCED_DIMENSION
#include "../../../g2g/scalar_vector_types.h"  // vec_type<T,N>
#include "../../../g2g/cuda/cuda_extra.h"      // index_x

namespace G2G {
#include "../../../g2g/cuda/gpu_variables.h"
#include "../../../g2g/cuda/kernels/energy.h"
}

using F4 = G2G::vec_type<float, 4>;

// ---------------------------------------------------------------------------
// Create a CUDA 2D texture from a row-major float[m*m] host array.
// tex2D(obj, col=j, row=i) returns rmm[i*m+j] = R[i][j].
// Caller must destroy texObj and free cuArray when done.
// ---------------------------------------------------------------------------
static cudaTextureObject_t make_rmm_texture(const std::vector<float>& rmm,
                                            int m, cudaArray_t& cuArray) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaMallocArray(&cuArray, &desc, m, m));
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, rmm.data(), m * sizeof(float),
                                 m * sizeof(float), m, cudaMemcpyHostToDevice));
  cudaResourceDesc resDesc{};
  resDesc.resType             = cudaResourceTypeArray;
  resDesc.res.array.array     = cuArray;
  cudaTextureDesc texDesc{};
  texDesc.readMode            = cudaReadModeElementType;
  texDesc.filterMode          = cudaFilterModePoint;
  texDesc.addressMode[0]      = cudaAddressModeClamp;
  texDesc.addressMode[1]      = cudaAddressModeClamp;
  cudaTextureObject_t texObj  = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
  return texObj;
}

// ---------------------------------------------------------------------------
// Run gpu_compute_density<float, lda=true> and return per-point densities.
// Allocates/frees all GPU resources internally.
// ---------------------------------------------------------------------------
static std::vector<float> run_density(const std::vector<float>& rmm, int m,
                                      const std::vector<float>& fv, int pts) {
  int n_rows   = (m + 2 * DENSITY_BLOCK_SIZE - 1) / (2 * DENSITY_BLOCK_SIZE);
  int out_size = n_rows * pts;

  float* d_fv;
  CUDA_CHECK(cudaMalloc(&d_fv, (size_t)m * pts * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_fv, fv.data(), (size_t)m * pts * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaArray_t         cuArray;
  cudaTextureObject_t texObj = make_rmm_texture(rmm, m, cuArray);

  float* d_pd;
  F4    *d_dxyz, *d_dd1, *d_dd2;
  CUDA_CHECK(cudaMalloc(&d_pd,   out_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dxyz, out_size * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dd1,  out_size * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dd2,  out_size * sizeof(F4)));

  dim3 block(DENSITY_BLOCK_SIZE);
  dim3 grid(pts, n_rows);
  G2G::gpu_compute_density<float, true><<<grid, block>>>(
      texObj, nullptr, nullptr, nullptr, pts,
      d_fv, nullptr, nullptr, m, d_pd, d_dxyz, d_dd1, d_dd2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_pd(out_size);
  CUDA_CHECK(cudaMemcpy(h_pd.data(), d_pd, out_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Sum block rows to get final density per point
  std::vector<float> density(pts, 0.f);
  for (int row = 0; row < n_rows; ++row)
    for (int p = 0; p < pts; ++p)
      density[p] += h_pd[row * pts + p];

  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);
  cudaFree(d_fv);
  cudaFree(d_pd); cudaFree(d_dxyz); cudaFree(d_dd1); cudaFree(d_dd2);
  return density;
}

// ---------------------------------------------------------------------------
// Double-precision CPU reference for FP accuracy testing.
// ---------------------------------------------------------------------------
static double ref_cpu_density_double(const std::vector<float>& rmm, int m,
                                     const std::vector<float>& fv, int point) {
  double rho = 0.0;
  for (int i = 0; i < m; ++i) {
    double w = 0.0;
    for (int j = 0; j <= i; ++j)
      w += (double)rmm[i * m + j] * (double)fv[m * point + j];
    rho += (double)fv[m * point + i] * w;
  }
  return rho;
}

// ============================================================================
int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_compute_density kernel");

  const float tol = 1e-4f;

  printf("[ lda=true: density only ]\n");

  // --- 1. m=1, pts=1: rho = R[0][0] * F[0]^2 = 2 * 9 = 18 ---
  {
    auto rho = run_density({2.f}, 1, {3.f}, 1);
    runner.check(fabsf(rho[0] - 18.f) < tol, "m=1 pts=1 trivial (rho=18)");
  }

  // --- 2. m=2, pts=1: hand-verify ---
  // R (lower tri): R[0][0]=1, R[1][0]=2, R[1][1]=3   F = [1, 2]
  // rho = F[0]*(R[0][0]*F[0]) + F[1]*(R[1][0]*F[0]+R[1][1]*F[1])
  //     = 1*1 + 2*(2+6) = 1 + 16 = 17
  {
    std::vector<float> rmm = {1.f, 0.f,  // row 0: R[0][0]=1, R[0][1]=0
                               2.f, 3.f}; // row 1: R[1][0]=2, R[1][1]=3
    auto rho = run_density(rmm, 2, {1.f, 2.f}, 1);
    runner.check(fabsf(rho[0] - 17.f) < tol, "m=2 pts=1 hand-verify (rho=17)");
  }

  // --- 3. m=4, pts=3 -> vs CPU ---
  {
    int m = 4, pts = 3;
    std::vector<float> rmm(m * m, 0.f), fv(m * pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i * m + j] = float(i * m + j + 1) * 0.1f;
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m * p + i] = float(p * m + i + 1) * 0.3f;
    auto got = run_density(rmm, m, fv, pts);
    bool ok = true;
    for (int p = 0; p < pts; ++p)
      ok &= fabsf(got[p] - ref_cpu_density(rmm, m, fv, p)) < tol;
    runner.check(ok, "m=4 pts=3 vs CPU");
  }

  // --- 4. m=100, pts=2: two block rows (100 > 64) ---
  {
    int m = 100, pts = 2;
    std::vector<float> rmm(m * m, 0.f), fv(m * pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i * m + j] = sinf(float(i * m + j) * 0.01f);
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m * p + i] = cosf(float(p * m + i) * 0.02f);
    auto got = run_density(rmm, m, fv, pts);
    bool ok = true;
    for (int p = 0; p < pts; ++p)
      ok &= fabsf(got[p] - ref_cpu_density(rmm, m, fv, p)) < 1e-2f;
    runner.check(ok, "m=100 pts=2 two block rows vs CPU");
  }

  // --- 5. m=130, pts=3: three block rows (130 > 128) ---
  {
    int m = 130, pts = 3;
    std::vector<float> rmm(m * m, 0.f), fv(m * pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i * m + j] = float(i + j + 1) * 0.005f;
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m * p + i] = float(p + i + 1) * 0.01f;
    auto got = run_density(rmm, m, fv, pts);
    bool ok = true;
    for (int p = 0; p < pts; ++p)
      ok &= fabsf(got[p] - ref_cpu_density(rmm, m, fv, p)) < 5e-2f;
    runner.check(ok, "m=130 pts=3 three block rows vs CPU");
  }

  printf("\n[ lda=false: density + gradient ]\n");

  // --- 6. m=2, pts=1, lda=false: analytical check ---
  // R[0][0]=2, R[1][0]=0, R[1][1]=1 (lower tri stored in texture)
  // F=[3,0],  Fg[0]=(1,0,0,0),  Fg[1]=(0,0,0,0)
  //
  // rho:  F[0]*(R[0][0]*F[0]) = 3*2*3 = 18  (F[1]=0 contributes nothing)
  // For i=0: w  = R[0][0]*F[0] = 6
  //          w3 = Fg[0]*R[0][0] = (1,0,0)*2 = (2,0,0)
  //          dxyz += Fg[0]*w + w3*F[0] = (1,0,0)*6 + (2,0,0)*3 = (12,0,0)
  //          dd1  += Fg[0]*w3*2 + 0 + 0 = (1,0,0)*(2,0,0)*2 = (4,0,0)
  {
    int m = 2, pts = 1;
    std::vector<float> rmm = {2.f, 0.f, 0.f, 1.f}; // R[0][0]=2, lower-tri
    std::vector<float> fv  = {3.f, 0.f};             // fv[m*p+i]
    // gradient_values[m*point + func]: layout size = m*pts
    std::vector<F4> gv = {F4(1.f, 0.f, 0.f, 0.f), F4(0.f, 0.f, 0.f, 0.f)};
    // hessian_values[m*2*point + 2*func + {0,1}]: size = m*2*pts
    std::vector<F4> hv(m * 2 * pts, F4(0.f, 0.f, 0.f, 0.f));

    int n_rows   = (m + 2 * DENSITY_BLOCK_SIZE - 1) / (2 * DENSITY_BLOCK_SIZE);
    int out_size = n_rows * pts;

    cudaArray_t         cuArray;
    cudaTextureObject_t texObj = make_rmm_texture(rmm, m, cuArray);

    float *d_fv; F4 *d_gv, *d_hv;
    CUDA_CHECK(cudaMalloc(&d_fv, m * pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gv, m * pts * sizeof(F4)));
    CUDA_CHECK(cudaMalloc(&d_hv, m * 2 * pts * sizeof(F4)));
    CUDA_CHECK(cudaMemcpy(d_fv, fv.data(), m * pts * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gv, gv.data(), m * pts * sizeof(F4),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hv, hv.data(), m * 2 * pts * sizeof(F4), cudaMemcpyHostToDevice));

    float *d_pd; F4 *d_dxyz, *d_dd1, *d_dd2;
    CUDA_CHECK(cudaMalloc(&d_pd,   out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dxyz, out_size * sizeof(F4)));
    CUDA_CHECK(cudaMalloc(&d_dd1,  out_size * sizeof(F4)));
    CUDA_CHECK(cudaMalloc(&d_dd2,  out_size * sizeof(F4)));

    dim3 block(DENSITY_BLOCK_SIZE);
    dim3 grid(pts, n_rows);
    G2G::gpu_compute_density<float, false><<<grid, block>>>(
        texObj, nullptr, nullptr, nullptr, pts,
        d_fv, d_gv, d_hv, m, d_pd, d_dxyz, d_dd1, d_dd2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_pd(out_size);
    std::vector<F4>    h_dxyz(out_size), h_dd1(out_size);
    CUDA_CHECK(cudaMemcpy(h_pd.data(),   d_pd,   out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dxyz.data(), d_dxyz, out_size * sizeof(F4),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dd1.data(),  d_dd1,  out_size * sizeof(F4),    cudaMemcpyDeviceToHost));

    // Sum block rows
    float rho_sum = 0.f;
    float dx = 0.f, dy = 0.f, dz = 0.f;
    float d1x = 0.f;
    for (int r = 0; r < n_rows; ++r) {
      rho_sum += h_pd  [r * pts];
      dx      += h_dxyz[r * pts].x;
      dy      += h_dxyz[r * pts].y;
      dz      += h_dxyz[r * pts].z;
      d1x     += h_dd1 [r * pts].x;
    }
    bool ok = fabsf(rho_sum - 18.f) < tol
           && fabsf(dx - 12.f) < tol && fabsf(dy) < tol && fabsf(dz) < tol
           && fabsf(d1x -  4.f) < tol;
    runner.check(ok, "lda=false m=2 pts=1: density + gradient");

    cudaDestroyTextureObject(texObj); cudaFreeArray(cuArray);
    cudaFree(d_fv); cudaFree(d_gv); cudaFree(d_hv);
    cudaFree(d_pd); cudaFree(d_dxyz); cudaFree(d_dd1); cudaFree(d_dd2);
  }

  printf("\n[ FP precision: Kahan accuracy ]\n");

  // --- 7. m=300, pts=1: Kahan precision test ---
  // With 300 basis functions, ~300 terms accumulated per thread.
  // Without Kahan: O(N*eps) ~ 3.6e-5 relative error.
  // With Kahan:    O(eps)   ~ 1.2e-7 relative error.
  // Compare GPU float result to double-precision CPU reference.
  {
    int m = 300, pts = 1;
    std::vector<float> rmm(m * m, 0.f), fv(m * pts);
    // Use realistic-scale values to stress FP accumulation
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i * m + j] = sinf(float(i * 7 + j * 3 + 1) * 0.0037f) * 0.1f;
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m * p + i] = cosf(float(i * 5 + 1) * 0.0051f) * 0.5f;

    double ref = ref_cpu_density_double(rmm, m, fv, 0);
    auto got = run_density(rmm, m, fv, pts);
    float rel_err = fabsf((float)(got[0] - ref) / (float)ref);
    printf("    m=300 ref=%.10g  gpu=%.10g  rel_err=%.2e\n",
           ref, (double)got[0], rel_err);
    // Kahan should give relative error < 1e-5 (much better than naive ~1e-2)
    runner.check(rel_err < 1e-5f, "m=300 Kahan precision vs double ref");
  }

  return runner.summary();
}
