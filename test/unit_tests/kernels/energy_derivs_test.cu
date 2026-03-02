// Unit tests for g2g/cuda/kernels/energy_derivs.h
//
// gpu_compute_density_derivs<scalar_type>:
//   For each basis function i at nucleus nuc[i]:
//     w = sum_k R[k][i] * F[k](p) * (i==k ? 2 : 1)
//     density_deriv[COALESCED_DIM(pts)*nuc[i]+p] -= Fgi * w
//
// function_values layout: fv[COALESCED_DIM(pts)*func + point]
// gradient_values layout: gv[COALESCED_DIM(pts)*func + point]
// RMM texture: tex2D(tex, col=i, row=k) = rmm[k*m+i] = R[k][i]
//
// Block = dim3(DENSITY_DERIV_BLOCK_SIZE=128), Grid = dim3(ceil(pts/128))
//
// Test coverage
// -------------
//   1. m=1, pts=1: trivial w = 2*R[0][0]*F[0]
//   2. m=2, pts=1, 2 atoms: hand-verify
//   3. m=4, pts=3: vs CPU reference
//   4. pts=200 > block size: multi-block vs CPU
//   5. Open-shell: Ra==Rb → deriv_a == deriv_b
//   6. Open-shell: asymmetric → vs CPU

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC   0

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "test_utils.h"
#include "../../../g2g/common.h"               // DENSITY_DERIV_BLOCK_SIZE, COALESCED_DIMENSION
#include "../../../g2g/matrix.h"               // COALESCED_DIMENSION macro
#include "../../../g2g/scalar_vector_types.h"  // vec_type<T,N>
#include "../../../g2g/cuda/cuda_extra.h"      // index_x

// fetch macro expected by energy_derivs.h (same as energy.h for FULL_DOUBLE=0)
#define fetch(t, x, y) tex2D<float>(t, x, y)

namespace G2G {
#include "../../../g2g/cuda/kernels/energy_derivs.h"
}

using F4 = G2G::vec_type<float, 4>;

// ---------------------------------------------------------------------------
// Create 2D CUDA texture from row-major float[m*m]
// tex2D(tex, col=j, row=i) = rmm[i*m+j] = R[i][j]
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
  texDesc.readMode       = cudaReadModeElementType;
  texDesc.filterMode     = cudaFilterModePoint;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
  return texObj;
}

// ---------------------------------------------------------------------------
// CPU reference
// fv layout: fv[COALESCED_DIM(pts)*func + point]
// gv layout: gv[COALESCED_DIM(pts)*func + point]
// nuc:       nuc[func] = nucleus index
// Output: deriv[COALESCED_DIM(pts)*nuc + point], initialized to 0
// Kernel: w_i = sum_k R[k][i]*fk*(k==i?2:1), deriv -= Fgi*w_i
// ---------------------------------------------------------------------------
static std::vector<F4> cpu_density_derivs(
    const std::vector<float>& rmm, int m,
    const std::vector<float>& fv, const std::vector<F4>& gv,
    const std::vector<unsigned>& nuc, int nuc_count, int pts) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4> deriv(cdim * nuc_count, F4(0.f, 0.f, 0.f, 0.f));
  for (int p = 0; p < pts; p++) {
    for (int i = 0; i < m; i++) {
      float w = 0.f;
      for (int k = 0; k < m; k++) {
        float fk = fv[cdim * k + p];
        float Rki = rmm[k * m + i];       // R[k][i] = tex2D(col=i, row=k)
        float factor = (i == k) ? 2.f : 1.f;
        w += Rki * fk * factor;
      }
      F4 Fgi = gv[cdim * i + p];
      int ni = (int)nuc[i];
      deriv[cdim * ni + p].x -= Fgi.x * w;
      deriv[cdim * ni + p].y -= Fgi.y * w;
      deriv[cdim * ni + p].z -= Fgi.z * w;
      deriv[cdim * ni + p].w -= Fgi.w * w;
    }
  }
  return deriv;
}

// ---------------------------------------------------------------------------
// Build fv/gv arrays in COALESCED_DIMENSION layout
// Input vals[func * pts + point] (row-major, func-major)
// ---------------------------------------------------------------------------
static std::vector<float> make_fv(int m, int pts,
                                   const std::vector<float>& vals) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<float> out(cdim * m, 0.f);
  for (int f = 0; f < m; f++)
    for (int p = 0; p < pts; p++)
      out[cdim * f + p] = vals[f * pts + p];
  return out;
}

static std::vector<F4> make_gv(int m, int pts,
                                const std::vector<F4>& vals) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4> out(cdim * m, F4(0.f, 0.f, 0.f, 0.f));
  for (int f = 0; f < m; f++)
    for (int p = 0; p < pts; p++)
      out[cdim * f + p] = vals[f * pts + p];
  return out;
}

// ---------------------------------------------------------------------------
// Run closed-shell kernel; returns deriv array [COALESCED_DIM(pts) * nuc_count]
// density_deriv is initialized to zero before the kernel runs.
// ---------------------------------------------------------------------------
static std::vector<F4> run_density_derivs(
    const std::vector<float>& rmm, int m,
    const std::vector<float>& fv, const std::vector<F4>& gv,
    const std::vector<unsigned>& nuc, int nuc_count, int pts) {
  int cdim = COALESCED_DIMENSION(pts);
  cudaArray_t cuArray;
  cudaTextureObject_t texObj = make_rmm_texture(rmm, m, cuArray);

  float    *d_fv;
  F4       *d_gv, *d_deriv;
  unsigned *d_nuc;
  CUDA_CHECK(cudaMalloc(&d_fv,    (size_t)cdim * m * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gv,    (size_t)cdim * m * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_nuc,   m * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_deriv, (size_t)cdim * nuc_count * sizeof(F4)));
  CUDA_CHECK(cudaMemcpy(d_fv,  fv.data(),  cdim * m * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gv,  gv.data(),  cdim * m * sizeof(F4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nuc, nuc.data(), m * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_deriv, 0, cdim * nuc_count * sizeof(F4)));

  dim3 block(DENSITY_DERIV_BLOCK_SIZE);
  dim3 grid((pts + DENSITY_DERIV_BLOCK_SIZE - 1) / DENSITY_DERIV_BLOCK_SIZE);
  G2G::gpu_compute_density_derivs<float><<<grid, block>>>(
      texObj, d_fv, d_gv, d_nuc, d_deriv, pts, m, nuc_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<F4> h_deriv(cdim * nuc_count);
  CUDA_CHECK(cudaMemcpy(h_deriv.data(), d_deriv, cdim * nuc_count * sizeof(F4),
                        cudaMemcpyDeviceToHost));

  cudaDestroyTextureObject(texObj); cudaFreeArray(cuArray);
  cudaFree(d_fv); cudaFree(d_gv); cudaFree(d_nuc); cudaFree(d_deriv);
  return h_deriv;
}

// ---------------------------------------------------------------------------
// Run open-shell kernel; returns {deriv_a, deriv_b}
// ---------------------------------------------------------------------------
static std::pair<std::vector<F4>, std::vector<F4>> run_density_derivs_open(
    const std::vector<float>& rmm_a, const std::vector<float>& rmm_b, int m,
    const std::vector<float>& fv, const std::vector<F4>& gv,
    const std::vector<unsigned>& nuc, int nuc_count, int pts) {
  int cdim = COALESCED_DIMENSION(pts);
  cudaArray_t ca_a, ca_b;
  cudaTextureObject_t tex_a = make_rmm_texture(rmm_a, m, ca_a);
  cudaTextureObject_t tex_b = make_rmm_texture(rmm_b, m, ca_b);

  float    *d_fv;
  F4       *d_gv, *d_da, *d_db;
  unsigned *d_nuc;
  CUDA_CHECK(cudaMalloc(&d_fv,  (size_t)cdim * m * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gv,  (size_t)cdim * m * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_nuc, m * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_da,  (size_t)cdim * nuc_count * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_db,  (size_t)cdim * nuc_count * sizeof(F4)));
  CUDA_CHECK(cudaMemcpy(d_fv,  fv.data(),  cdim * m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gv,  gv.data(),  cdim * m * sizeof(F4),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nuc, nuc.data(), m * sizeof(unsigned),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_da, 0, cdim * nuc_count * sizeof(F4)));
  CUDA_CHECK(cudaMemset(d_db, 0, cdim * nuc_count * sizeof(F4)));

  dim3 block(DENSITY_DERIV_BLOCK_SIZE);
  dim3 grid((pts + DENSITY_DERIV_BLOCK_SIZE - 1) / DENSITY_DERIV_BLOCK_SIZE);
  G2G::gpu_compute_density_derivs_open<float><<<grid, block>>>(
      tex_a, tex_b, d_fv, d_gv, d_nuc, d_da, d_db, pts, m, nuc_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<F4> ha(cdim * nuc_count), hb(cdim * nuc_count);
  CUDA_CHECK(cudaMemcpy(ha.data(), d_da, cdim * nuc_count * sizeof(F4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(), d_db, cdim * nuc_count * sizeof(F4),
                        cudaMemcpyDeviceToHost));

  cudaDestroyTextureObject(tex_a); cudaFreeArray(ca_a);
  cudaDestroyTextureObject(tex_b); cudaFreeArray(ca_b);
  cudaFree(d_fv); cudaFree(d_gv); cudaFree(d_nuc); cudaFree(d_da); cudaFree(d_db);
  return {ha, hb};
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------
static bool near4(F4 a, F4 b, float tol = 1e-4f) {
  return fabsf(a.x-b.x)<=tol && fabsf(a.y-b.y)<=tol &&
         fabsf(a.z-b.z)<=tol && fabsf(a.w-b.w)<=tol;
}

// Compare valid [nuc, point] entries; padding elements are ignored.
static bool all_near_derivs(const std::vector<F4>& got,
                             const std::vector<F4>& ref,
                             int nuc_count, int pts, float tol = 1e-4f) {
  int cdim = COALESCED_DIMENSION(pts);
  for (int na = 0; na < nuc_count; na++)
    for (int p = 0; p < pts; p++)
      if (!near4(got[cdim*na+p], ref[cdim*na+p], tol)) return false;
  return true;
}

// ============================================================================
int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_compute_density_derivs kernel");

  const float tol = 1e-4f;

  printf("[ closed-shell ]\n");

  // --- 1. m=1, pts=1: trivial ---
  // R=[2], fv=[3], gv=[(1,0,0,0)], nuc=[0]
  // w = R[0][0]*3*2 = 12, deriv[atom0] = -(1,0,0)*12 = (-12,0,0,0)
  {
    std::vector<float> rmm = {2.f};
    auto fv = make_fv(1, 1, {3.f});
    auto gv = make_gv(1, 1, {F4(1.f, 0.f, 0.f, 0.f)});
    auto got = run_density_derivs(rmm, 1, fv, gv, {0u}, 1, 1);
    int cdim = COALESCED_DIMENSION(1);
    runner.check(near4(got[cdim*0+0], F4(-12.f, 0.f, 0.f, 0.f), tol),
                 "m=1 pts=1 trivial");
  }

  // --- 2. m=2, pts=1, 2 atoms: hand-verify ---
  // rmm = {1,0,2,3} → R[0][0]=1, R[1][0]=2, R[1][1]=3
  // fv=[1,2], gv=[(1,0,0,0),(0,1,0,0)], nuc=[0,1]
  // i=0: w = R[0][0]*1*2 + R[1][0]*2*1 = 2+4 = 6
  //        deriv[atom0] -= (1,0,0,0)*6 = (-6,0,0,0)
  // i=1: w = R[0][1]*1*1 + R[1][1]*2*2 = 0+12 = 12
  //        deriv[atom1] -= (0,1,0,0)*12 = (0,-12,0,0)
  {
    std::vector<float> rmm = {1.f, 0.f, 2.f, 3.f};
    auto fv = make_fv(2, 1, {1.f, 2.f});
    auto gv = make_gv(2, 1, {F4(1.f, 0.f, 0.f, 0.f), F4(0.f, 1.f, 0.f, 0.f)});
    auto got = run_density_derivs(rmm, 2, fv, gv, {0u, 1u}, 2, 1);
    int cdim = COALESCED_DIMENSION(1);
    bool ok = near4(got[cdim*0+0], F4(-6.f,  0.f,  0.f, 0.f), tol) &&
              near4(got[cdim*1+0], F4( 0.f, -12.f, 0.f, 0.f), tol);
    runner.check(ok, "m=2 pts=1 two atoms hand-verify");
  }

  // --- 3. m=4, pts=3: vs CPU ---
  {
    int m = 4, pts = 3, nuc_count = 2;
    std::vector<float> rmm(m * m, 0.f);
    for (int i = 0; i < m; i++)
      for (int j = 0; j <= i; j++)
        rmm[i * m + j] = float(i * m + j + 1) * 0.1f;
    std::vector<float> fv_raw(m * pts);
    std::vector<F4>    gv_raw(m * pts);
    std::vector<unsigned> nuc = {0u, 0u, 1u, 1u};
    for (int f = 0; f < m; f++)
      for (int p = 0; p < pts; p++) {
        fv_raw[f * pts + p] = float(f * pts + p + 1) * 0.3f;
        gv_raw[f * pts + p] = F4(float(f+p)*0.1f, float(p-f)*0.1f,
                                  float(f)*0.05f, 0.f);
      }
    auto fv  = make_fv(m, pts, fv_raw);
    auto gv  = make_gv(m, pts, gv_raw);
    auto got = run_density_derivs(rmm, m, fv, gv, nuc, nuc_count, pts);
    auto ref = cpu_density_derivs(rmm, m, fv, gv, nuc, nuc_count, pts);
    runner.check(all_near_derivs(got, ref, nuc_count, pts, tol),
                 "m=4 pts=3 vs CPU");
  }

  // --- 4. pts=200: two thread blocks vs CPU ---
  {
    int m = 5, pts = 200, nuc_count = 2;
    std::vector<float> rmm(m * m, 0.f);
    for (int i = 0; i < m; i++)
      for (int j = 0; j <= i; j++)
        rmm[i * m + j] = sinf(float(i * m + j) * 0.2f);
    std::vector<float> fv_raw(m * pts);
    std::vector<F4>    gv_raw(m * pts);
    std::vector<unsigned> nuc = {0u, 0u, 0u, 1u, 1u};
    for (int f = 0; f < m; f++)
      for (int p = 0; p < pts; p++) {
        fv_raw[f * pts + p] = cosf(float(f * pts + p) * 0.01f);
        gv_raw[f * pts + p] = F4(sinf(float(f+p)*0.05f),
                                  cosf(float(f+p)*0.03f), 0.f, 0.f);
      }
    auto fv  = make_fv(m, pts, fv_raw);
    auto gv  = make_gv(m, pts, gv_raw);
    auto got = run_density_derivs(rmm, m, fv, gv, nuc, nuc_count, pts);
    auto ref = cpu_density_derivs(rmm, m, fv, gv, nuc, nuc_count, pts);
    runner.check(all_near_derivs(got, ref, nuc_count, pts, 5e-4f),
                 "pts=200 multi-block vs CPU");
  }

  printf("\n[ open-shell ]\n");

  // --- 5. Open: Ra==Rb → deriv_a == deriv_b ---
  {
    int m = 4, pts = 5, nuc_count = 2;
    std::vector<float> rmm(m * m, 0.f);
    for (int i = 0; i < m; i++)
      for (int j = 0; j <= i; j++)
        rmm[i * m + j] = float(i + j + 1) * 0.15f;
    std::vector<float> fv_raw(m * pts);
    std::vector<F4>    gv_raw(m * pts);
    std::vector<unsigned> nuc = {0u, 0u, 1u, 1u};
    for (int f = 0; f < m; f++)
      for (int p = 0; p < pts; p++) {
        fv_raw[f * pts + p] = float(f * pts + p + 1) * 0.1f;
        gv_raw[f * pts + p] = F4(float(f+1)*0.2f, float(p+1)*0.1f, 0.f, 0.f);
      }
    auto fv = make_fv(m, pts, fv_raw);
    auto gv = make_gv(m, pts, gv_raw);
    auto pair1 = run_density_derivs_open(rmm, rmm, m, fv, gv,
                                          nuc, nuc_count, pts);
    runner.check(all_near_derivs(pair1.first, pair1.second, nuc_count, pts, tol),
                 "open-shell Ra==Rb → deriv_a == deriv_b");
  }

  // --- 6. Open: asymmetric → vs CPU ---
  {
    int m = 4, pts = 6, nuc_count = 2;
    std::vector<float> rmm_a(m * m, 0.f), rmm_b(m * m, 0.f);
    for (int i = 0; i < m; i++)
      for (int j = 0; j <= i; j++) {
        rmm_a[i * m + j] = float(i * m + j + 1) * 0.1f;
        rmm_b[i * m + j] = float(i + j + 2) * 0.07f;
      }
    std::vector<float> fv_raw(m * pts);
    std::vector<F4>    gv_raw(m * pts);
    std::vector<unsigned> nuc = {0u, 0u, 1u, 1u};
    for (int f = 0; f < m; f++)
      for (int p = 0; p < pts; p++) {
        fv_raw[f * pts + p] = sinf(float(f * pts + p) * 0.1f);
        gv_raw[f * pts + p] = F4(float(f)*0.3f, float(p)*0.2f,
                                  float(f+p)*0.1f, 0.f);
      }
    auto fv   = make_fv(m, pts, fv_raw);
    auto gv   = make_gv(m, pts, gv_raw);
    auto pair2 = run_density_derivs_open(rmm_a, rmm_b, m, fv, gv,
                                          nuc, nuc_count, pts);
    auto ref_a = cpu_density_derivs(rmm_a, m, fv, gv, nuc, nuc_count, pts);
    auto ref_b = cpu_density_derivs(rmm_b, m, fv, gv, nuc, nuc_count, pts);
    bool ok = all_near_derivs(pair2.first,  ref_a, nuc_count, pts, tol) &&
              all_near_derivs(pair2.second, ref_b, nuc_count, pts, tol);
    runner.check(ok, "open-shell asymmetric → vs CPU");
  }

  return runner.summary();
}
