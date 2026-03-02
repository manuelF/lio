// Unit tests for g2g/cuda/kernels/accumulate_point.h
//
// gpu_accumulate_point (closed-shell) accumulates density/gradient rows and
// calls calc_ggaCS_in to compute XC energy density (exc) and potential (y2a):
//
//   For each point p:
//     rho  = sum_{row=0}^{block_height-1}  partial_density[row*points + p]
//     grad = sum  dxyz[row*points + p]
//     lap1 = sum  dd1[row*points + p]
//     lap2 = sum  dd2[row*points + p]
//     calc_ggaCS_in(rho, grad, lap1, lap2, exc_x, exc_c, y2a, iexch=9)
//     energy[p] = rho * point_weight[p] * (exc_x + exc_c)   [if compute_energy]
//     factor[p] = point_weight[p] * y2a                     [if compute_factor]
//
// gpu_accumulate_point_open (open-shell) is the same but with separate
// alpha/beta densities and calls calc_ggaOS instead.
//
// Tests
// -----
//   1. Closed: block_height=1, varied rho (zero gradient) — compute_energy
//   2. Closed: block_height=1, varied rho (zero gradient) — compute_factor
//   3. Closed: block_height=1, both flags true
//   4. Closed: block_height=2 — row accumulation
//   5. Closed: points=200 (not multiple of DENSITY_ACCUM_BLOCK_SIZE=128)
//   6. Closed: density below threshold → energy=0, factor=0
//   7. Open: block_height=1, varied rho_a=rho_b, zero gradients

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC   0

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

#include "test_utils.h"
#include "../../../g2g/common.h"              // DENSITY_ACCUM_BLOCK_SIZE

// calc_ggaCS.h and calc_ggaOS.h open their own namespace G2G {} blocks,
// so they must be included OUTSIDE any enclosing namespace.
#include "../../../g2g/pointxc/calc_ggaCS.h"
#include "../../../g2g/pointxc/calc_ggaOS.h"

#include "../../../g2g/scalar_vector_types.h" // vec_type<T,N>
#include "../../../g2g/cuda/cuda_extra.h"     // index_x(), etc.

namespace G2G {
#include "../../../g2g/cuda/kernels/accumulate_point.h"
}

// ---------------------------------------------------------------------------
// Shorthand type aliases.
// ---------------------------------------------------------------------------
using F4 = G2G::vec_type<float, 4>;

// ---------------------------------------------------------------------------
// CPU reference for closed-shell gpu_accumulate_point.
//   pd[row*points+p], dxyz[row*points+p], dd1[row*points+p], dd2[row*points+p]
//   Returns energy and factor vectors (points elements each).
// ---------------------------------------------------------------------------
static void cpu_accumulate_closed(
    const std::vector<float>& pw,        // point weights [points]
    uint points, int block_height,
    const std::vector<float>& pd,        // partial_density [block_height*points]
    const std::vector<F4>&   dxyz_v,    // [block_height*points]
    const std::vector<F4>&   dd1_v,     // [block_height*points]
    const std::vector<F4>&   dd2_v,     // [block_height*points]
    std::vector<float>& energy_ref,
    std::vector<float>& factor_ref)
{
  energy_ref.resize(points, 0.f);
  factor_ref.resize(points, 0.f);

  for (uint p = 0; p < points; p++) {
    float sum_dens = 0.f;
    F4 sum_dxyz(0.f,0.f,0.f,0.f), sum_dd1(0.f,0.f,0.f,0.f),
       sum_dd2(0.f,0.f,0.f,0.f);

    for (int j = 0; j < block_height; j++) {
      int idx = j * int(points) + int(p);
      sum_dens   += pd[idx];
      // Manual component-wise add (vec_type += uses float4 + and copy-ctor)
      sum_dxyz.x += dxyz_v[idx].x; sum_dxyz.y += dxyz_v[idx].y;
      sum_dxyz.z += dxyz_v[idx].z; sum_dxyz.w += dxyz_v[idx].w;
      sum_dd1.x  += dd1_v[idx].x;  sum_dd1.y  += dd1_v[idx].y;
      sum_dd1.z  += dd1_v[idx].z;  sum_dd1.w  += dd1_v[idx].w;
      sum_dd2.x  += dd2_v[idx].x;  sum_dd2.y  += dd2_v[idx].y;
      sum_dd2.z  += dd2_v[idx].z;  sum_dd2.w  += dd2_v[idx].w;
    }

    float exc_x = 0.f, exc_c = 0.f, y2a = 0.f;
    G2G::calc_ggaCS_in<float, 4>(sum_dens, sum_dxyz, sum_dd1, sum_dd2,
                                  exc_x, exc_c, y2a, 9);

    energy_ref[p] = (sum_dens * pw[p]) * (exc_x + exc_c);
    factor_ref[p] = pw[p] * y2a;
  }
}

// ---------------------------------------------------------------------------
// CPU reference for open-shell gpu_accumulate_point_open.
// ---------------------------------------------------------------------------
static void cpu_accumulate_open(
    const std::vector<float>& pw, uint points, int block_height,
    const std::vector<float>& pd_a,  const std::vector<float>& pd_b,
    const std::vector<F4>&    dxyz_a, const std::vector<F4>&   dxyz_b,
    const std::vector<F4>&    dd1_a,  const std::vector<F4>&   dd1_b,
    const std::vector<F4>&    dd2_a,  const std::vector<F4>&   dd2_b,
    std::vector<float>& energy_ref,
    std::vector<float>& factor_a_ref,
    std::vector<float>& factor_b_ref)
{
  energy_ref.resize(points, 0.f);
  factor_a_ref.resize(points, 0.f);
  factor_b_ref.resize(points, 0.f);

  for (uint p = 0; p < points; p++) {
    float sa = 0.f, sb = 0.f;
    F4 da(0,0,0,0), db(0,0,0,0), h1a(0,0,0,0), h1b(0,0,0,0),
       h2a(0,0,0,0), h2b(0,0,0,0);

    for (int j = 0; j < block_height; j++) {
      int idx = j * int(points) + int(p);
      sa    += pd_a[idx];
      sb    += pd_b[idx];
      da.x  += dxyz_a[idx].x; da.y  += dxyz_a[idx].y;
      da.z  += dxyz_a[idx].z; da.w  += dxyz_a[idx].w;
      db.x  += dxyz_b[idx].x; db.y  += dxyz_b[idx].y;
      db.z  += dxyz_b[idx].z; db.w  += dxyz_b[idx].w;
      h1a.x += dd1_a[idx].x;  h1a.y += dd1_a[idx].y;
      h1a.z += dd1_a[idx].z;  h1a.w += dd1_a[idx].w;
      h1b.x += dd1_b[idx].x;  h1b.y += dd1_b[idx].y;
      h1b.z += dd1_b[idx].z;  h1b.w += dd1_b[idx].w;
      h2a.x += dd2_a[idx].x;  h2a.y += dd2_a[idx].y;
      h2a.z += dd2_a[idx].z;  h2a.w += dd2_a[idx].w;
      h2b.x += dd2_b[idx].x;  h2b.y += dd2_b[idx].y;
      h2b.z += dd2_b[idx].z;  h2b.w += dd2_b[idx].w;
    }

    float exc_corr, exc, corr, corr1, corr2, v_a, v_b;
    G2G::calc_ggaOS<float, 4>(sa, sb, da, db, h1a, h1b, h2a, h2b,
                               exc_corr, exc, corr, corr1, corr2, v_a, v_b, 9);

    energy_ref[p]   = ((sa + sb) * pw[p]) * exc_corr;
    factor_a_ref[p] = pw[p] * v_a;
    factor_b_ref[p] = pw[p] * v_b;
  }
}

// ---------------------------------------------------------------------------
// Run closed-shell kernel and compare energy/factor against CPU reference.
// ---------------------------------------------------------------------------
template<bool CE, bool CF>
static bool run_closed_test(uint points, int block_height,
                             const std::vector<float>& pw,
                             const std::vector<float>& pd,
                             const std::vector<F4>&   dxyz_v,
                             const std::vector<F4>&   dd1_v,
                             const std::vector<F4>&   dd2_v,
                             float tol = 1e-5f)
{
  // CPU reference
  std::vector<float> energy_ref, factor_ref;
  cpu_accumulate_closed(pw, points, block_height, pd, dxyz_v, dd1_v, dd2_v,
                        energy_ref, factor_ref);

  // Device allocations
  float* d_energy  = nullptr;
  float* d_factor  = nullptr;
  float* d_pw      = nullptr;
  float* d_pd      = nullptr;
  F4*    d_dxyz    = nullptr;
  F4*    d_dd1     = nullptr;
  F4*    d_dd2     = nullptr;

  size_t sz_row = (size_t)block_height * points;

  CUDA_CHECK(cudaMalloc(&d_energy, points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_factor, points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pw,     points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pd,     sz_row * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dxyz,   sz_row * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dd1,    sz_row * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_dd2,    sz_row * sizeof(F4)));

  CUDA_CHECK(cudaMemset(d_energy, 0, points * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_factor, 0, points * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_pw,   pw.data(),     points * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pd,   pd.data(),     sz_row * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dxyz, dxyz_v.data(), sz_row * sizeof(F4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dd1,  dd1_v.data(),  sz_row * sizeof(F4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dd2,  dd2_v.data(),  sz_row * sizeof(F4),
                        cudaMemcpyHostToDevice));

  dim3 block(DENSITY_ACCUM_BLOCK_SIZE);
  dim3 grid((points + DENSITY_ACCUM_BLOCK_SIZE - 1) / DENSITY_ACCUM_BLOCK_SIZE);
  G2G::gpu_accumulate_point<float, CE, CF, false><<<grid, block>>>(
      d_energy, d_factor, d_pw, points, block_height,
      d_pd, d_dxyz, d_dd1, d_dd2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_energy(points), h_factor(points);
  CUDA_CHECK(cudaMemcpy(h_energy.data(), d_energy,
                        points * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_factor.data(), d_factor,
                        points * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_energy); cudaFree(d_factor); cudaFree(d_pw);
  cudaFree(d_pd);     cudaFree(d_dxyz);  cudaFree(d_dd1); cudaFree(d_dd2);

  bool ok = true;
  for (uint p = 0; p < points; p++) {
    if (CE) {
      float got = h_energy[p], exp = energy_ref[p];
      if (std::abs(got - exp) > tol * (1.f + std::abs(exp))) {
        printf("    energy MISMATCH p=%u: expected %.6g  got %.6g\n",
               p, (double)exp, (double)got);
        ok = false;
        break;
      }
    }
    if (CF) {
      float got = h_factor[p], exp = factor_ref[p];
      if (std::abs(got - exp) > tol * (1.f + std::abs(exp))) {
        printf("    factor MISMATCH p=%u: expected %.6g  got %.6g\n",
               p, (double)exp, (double)got);
        ok = false;
        break;
      }
    }
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Helper: build closed-shell input with varied density, zero gradients.
// ---------------------------------------------------------------------------
static void make_closed_input(uint points, int block_height,
                               float rho_scale,
                               std::vector<float>& pw,
                               std::vector<float>& pd,
                               std::vector<F4>&   dxyz_v,
                               std::vector<F4>&   dd1_v,
                               std::vector<F4>&   dd2_v)
{
  pw.resize(points);
  pd.resize((size_t)block_height * points, 0.f);
  dxyz_v.assign((size_t)block_height * points, F4(0,0,0,0));
  dd1_v .assign((size_t)block_height * points, F4(0,0,0,0));
  dd2_v .assign((size_t)block_height * points, F4(0,0,0,0));

  for (uint p = 0; p < points; p++) {
    pw[p] = float(p + 1) * 0.001f;
    for (int j = 0; j < block_height; j++) {
      // Vary density across rows: (row+1)*(p+1)*scale
      pd[(size_t)j * points + p] = float((j+1)*(p+1)) * rho_scale;
    }
  }
}

// ---------------------------------------------------------------------------
// Run open-shell kernel and compare against CPU reference.
// ---------------------------------------------------------------------------
template<bool CE, bool CF>
static bool run_open_test(uint points, int block_height,
                           const std::vector<float>& pw,
                           const std::vector<float>& pd_a,
                           const std::vector<float>& pd_b,
                           const std::vector<F4>&   dxyz_a,
                           const std::vector<F4>&   dxyz_b,
                           const std::vector<F4>&   dd1_a,
                           const std::vector<F4>&   dd1_b,
                           const std::vector<F4>&   dd2_a,
                           const std::vector<F4>&   dd2_b,
                           float tol = 1e-5f)
{
  std::vector<float> energy_ref, fa_ref, fb_ref;
  cpu_accumulate_open(pw, points, block_height,
                      pd_a, pd_b, dxyz_a, dxyz_b, dd1_a, dd1_b, dd2_a, dd2_b,
                      energy_ref, fa_ref, fb_ref);

  size_t sz_row = (size_t)block_height * points;

  float* d_energy  = nullptr;
  float* d_ei      = nullptr;
  float* d_ec      = nullptr;
  float* d_ec1     = nullptr;
  float* d_ec2     = nullptr;
  float* d_fa      = nullptr;
  float* d_fb      = nullptr;
  float* d_pw      = nullptr;
  float* d_pda     = nullptr;
  float* d_pdb     = nullptr;
  F4*    d_dxyz_a  = nullptr;
  F4*    d_dxyz_b  = nullptr;
  F4*    d_dd1_a   = nullptr;
  F4*    d_dd1_b   = nullptr;
  F4*    d_dd2_a   = nullptr;
  F4*    d_dd2_b   = nullptr;

  auto alloc_p  = [&](float** p) { CUDA_CHECK(cudaMalloc(p, points*sizeof(float))); CUDA_CHECK(cudaMemset(*p, 0, points*sizeof(float))); };
  auto alloc_r  = [&](float** p) { CUDA_CHECK(cudaMalloc(p, sz_row*sizeof(float))); };
  auto alloc_r4 = [&](F4** p)   { CUDA_CHECK(cudaMalloc(p, sz_row*sizeof(F4))); };

  alloc_p(&d_energy); alloc_p(&d_ei); alloc_p(&d_ec);
  alloc_p(&d_ec1);    alloc_p(&d_ec2);
  alloc_p(&d_fa);     alloc_p(&d_fb); alloc_p(&d_pw);
  alloc_r(&d_pda);    alloc_r(&d_pdb);
  alloc_r4(&d_dxyz_a); alloc_r4(&d_dxyz_b);
  alloc_r4(&d_dd1_a);  alloc_r4(&d_dd1_b);
  alloc_r4(&d_dd2_a);  alloc_r4(&d_dd2_b);

  auto copy_p  = [&](float* d, const std::vector<float>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), points*sizeof(float), cudaMemcpyHostToDevice)); };
  auto copy_r  = [&](float* d, const std::vector<float>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), sz_row*sizeof(float), cudaMemcpyHostToDevice)); };
  auto copy_r4 = [&](F4* d, const std::vector<F4>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), sz_row*sizeof(F4), cudaMemcpyHostToDevice)); };

  copy_p(d_pw, pw);
  copy_r(d_pda, pd_a);    copy_r(d_pdb, pd_b);
  copy_r4(d_dxyz_a, dxyz_a); copy_r4(d_dxyz_b, dxyz_b);
  copy_r4(d_dd1_a,  dd1_a);  copy_r4(d_dd1_b,  dd1_b);
  copy_r4(d_dd2_a,  dd2_a);  copy_r4(d_dd2_b,  dd2_b);

  dim3 block(DENSITY_ACCUM_BLOCK_SIZE);
  dim3 grid((points + DENSITY_ACCUM_BLOCK_SIZE - 1) / DENSITY_ACCUM_BLOCK_SIZE);
  G2G::gpu_accumulate_point_open<float, CE, CF, false><<<grid, block>>>(
      d_energy, d_ei, d_ec, d_ec1, d_ec2,
      d_fa, d_fb,
      d_pw, points, block_height,
      d_pda, d_dxyz_a, d_dd1_a, d_dd2_a,
      d_pdb, d_dxyz_b, d_dd1_b, d_dd2_b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_energy(points), h_fa(points), h_fb(points);
  CUDA_CHECK(cudaMemcpy(h_energy.data(), d_energy, points*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_fa.data(),     d_fa,     points*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_fb.data(),     d_fb,     points*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_energy); cudaFree(d_ei); cudaFree(d_ec); cudaFree(d_ec1); cudaFree(d_ec2);
  cudaFree(d_fa); cudaFree(d_fb); cudaFree(d_pw);
  cudaFree(d_pda); cudaFree(d_pdb);
  cudaFree(d_dxyz_a); cudaFree(d_dxyz_b);
  cudaFree(d_dd1_a); cudaFree(d_dd1_b);
  cudaFree(d_dd2_a); cudaFree(d_dd2_b);

  bool ok = true;
  for (uint p = 0; p < points; p++) {
    if (CE) {
      float got = h_energy[p], exp = energy_ref[p];
      if (std::abs(got - exp) > tol * (1.f + std::abs(exp))) {
        printf("    open energy MISMATCH p=%u: expected %.6g  got %.6g\n",
               p, (double)exp, (double)got);
        ok = false; break;
      }
    }
    if (CF) {
      float ga = h_fa[p], ea = fa_ref[p];
      float gb = h_fb[p], eb = fb_ref[p];
      if (std::abs(ga - ea) > tol * (1.f + std::abs(ea)) ||
          std::abs(gb - eb) > tol * (1.f + std::abs(eb))) {
        printf("    open factor MISMATCH p=%u: fa=%.6g/%.6g  fb=%.6g/%.6g\n",
               p, (double)ea, (double)ga, (double)eb, (double)gb);
        ok = false; break;
      }
    }
  }
  return ok;
}

int main()
{
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_accumulate_point kernel");

  // --- Closed-shell tests ---
  printf("[ gpu_accumulate_point (closed-shell) ]\n");

  // Test 1: compute_energy only, block_height=1
  {
    uint N = 128;  // one full block
    std::vector<float> pw, pd;
    std::vector<F4> dxyz, dd1, dd2;
    make_closed_input(N, 1, 0.01f, pw, pd, dxyz, dd1, dd2);
    runner.check(run_closed_test<true, false>(N, 1, pw, pd, dxyz, dd1, dd2),
                 "float  pts=128 bh=1  compute_energy only");
  }

  // Test 2: compute_factor only, block_height=1
  {
    uint N = 128;
    std::vector<float> pw, pd;
    std::vector<F4> dxyz, dd1, dd2;
    make_closed_input(N, 1, 0.01f, pw, pd, dxyz, dd1, dd2);
    runner.check(run_closed_test<false, true>(N, 1, pw, pd, dxyz, dd1, dd2),
                 "float  pts=128 bh=1  compute_factor only");
  }

  // Test 3: both flags, block_height=1
  {
    uint N = 128;
    std::vector<float> pw, pd;
    std::vector<F4> dxyz, dd1, dd2;
    make_closed_input(N, 1, 0.01f, pw, pd, dxyz, dd1, dd2);
    runner.check(run_closed_test<true, true>(N, 1, pw, pd, dxyz, dd1, dd2),
                 "float  pts=128 bh=1  energy+factor");
  }

  // Test 4: block_height=3, row accumulation
  {
    uint N = 128;
    std::vector<float> pw, pd;
    std::vector<F4> dxyz, dd1, dd2;
    make_closed_input(N, 3, 0.005f, pw, pd, dxyz, dd1, dd2);
    runner.check(run_closed_test<true, true>(N, 3, pw, pd, dxyz, dd1, dd2),
                 "float  pts=128 bh=3  row accumulation");
  }

  // Test 5: points=200 (spans two blocks, partial second block)
  {
    uint N = 200;
    std::vector<float> pw, pd;
    std::vector<F4> dxyz, dd1, dd2;
    make_closed_input(N, 1, 0.01f, pw, pd, dxyz, dd1, dd2);
    runner.check(run_closed_test<true, true>(N, 1, pw, pd, dxyz, dd1, dd2),
                 "float  pts=200 bh=1  partial last block");
  }

  // Test 6: density below threshold → energy=0, factor=0
  {
    uint N = 64;
    std::vector<float> pw(N, 0.1f), pd(N, 1e-20f); // below MINIMUM_DENSITY_VALUE
    std::vector<F4> dxyz(N, F4(0,0,0,0)), dd1(N, F4(0,0,0,0)), dd2(N, F4(0,0,0,0));
    // CPU reference will also return 0 since calc_ggaCS_in returns 0 for low dens.
    runner.check(run_closed_test<true, true>(N, 1, pw, pd, dxyz, dd1, dd2, 1e-7f),
                 "float  pts=64  bh=1  density below threshold → zero output");
  }

  // --- Open-shell tests ---
  printf("\n[ gpu_accumulate_point_open (open-shell) ]\n");

  // Test 7: equal alpha=beta, block_height=1
  {
    uint N = 128;
    std::vector<float> pw(N), pd_a(N), pd_b(N);
    std::vector<F4> da(N, F4(0,0,0,0)), db(N, F4(0,0,0,0)),
                    h1a(N, F4(0,0,0,0)), h1b(N, F4(0,0,0,0)),
                    h2a(N, F4(0,0,0,0)), h2b(N, F4(0,0,0,0));
    for (uint p = 0; p < N; p++) {
      pw[p]  = float(p+1) * 0.001f;
      pd_a[p] = float(p+1) * 0.005f;
      pd_b[p] = pd_a[p];  // equal spins
    }
    runner.check(run_open_test<true, true>(N, 1, pw, pd_a, pd_b,
                                           da, db, h1a, h1b, h2a, h2b),
                 "float  pts=128 bh=1  open-shell alpha=beta, energy+factor");
  }

  // Test 8: asymmetric alpha≠beta, block_height=2
  {
    uint N = 100;
    std::vector<float> pw(N), pd_a((size_t)2*N), pd_b((size_t)2*N);
    std::vector<F4> da((size_t)2*N, F4(0,0,0,0)), db((size_t)2*N, F4(0,0,0,0)),
                    h1a((size_t)2*N, F4(0,0,0,0)), h1b((size_t)2*N, F4(0,0,0,0)),
                    h2a((size_t)2*N, F4(0,0,0,0)), h2b((size_t)2*N, F4(0,0,0,0));
    for (uint p = 0; p < N; p++) {
      pw[p] = float(p+1) * 0.001f;
      for (int j = 0; j < 2; j++) {
        pd_a[(size_t)j*N + p] = float((j+1)*(p+1)) * 0.003f;
        pd_b[(size_t)j*N + p] = float((j+1)*(p+1)) * 0.002f;
      }
    }
    runner.check(run_open_test<true, true>(N, 2, pw, pd_a, pd_b,
                                           da, db, h1a, h1b, h2a, h2b),
                 "float  pts=100 bh=2  open-shell alpha≠beta, energy+factor");
  }

  return runner.summary();
}
