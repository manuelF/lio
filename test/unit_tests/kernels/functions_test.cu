// Unit tests for g2g/cuda/kernels/functions.h
//
// gpu_compute_functions<scalar_type, do_forces=false, do_gga=false>:
//   Evaluates Gaussian-type orbital (GTO) basis functions on a grid.
//
//   S-shell: phi = t  where t = sum_c coeff_c * exp(-alpha_c * |r - R_A|^2)
//   P-shell: phi_Px = vx*t,  phi_Py = vy*t,  phi_Pz = vz*t
//   D-shell: phi_XX = norm*vx^2*t,  phi_XY = vy*vx*t,  phi_YY = norm*vy^2*t
//            phi_XZ = vz*vx*t,      phi_YZ = vz*vy*t,  phi_ZZ = norm*vz^2*t
//   where v = r - R_A  and  norm = gpu_normalization_factor
//
// D-shell ordering (functions.z): 0=XX, 1=YX, 2=YY, 3=ZX, 4=ZY, 5=ZZ
//
// Output layout: function_values[COALESCED_DIM(pts)*func + point]
// factor_ac layout: factor_ac[COALESCED_DIM(total_funcs)*contraction + func]
//                   with .x=exponent, .y=coefficient
//
// Atom positions set via cudaMemcpyToSymbol(G2G::gpu_atom_positions, ...)
// Normalization factor set via cudaMemcpyToSymbol(G2G::gpu_normalization_factor, ...)
//
// Block = dim3(FUNCTIONS_BLOCK_SIZE=128), Grid = dim3(ceil(pts/128))
//
// Test coverage
// -------------
//   1. 1 S-function, 1 contraction, point at atom center → phi = coeff
//   2. 1 S-function, 1 contraction, point off-center → phi = coeff*exp(-a*r^2)
//   3. 1 S-function, 2 contractions → phi = sum of contributions
//   4. 1 P-shell (3 functions) → Px=vx*t, Py=vy*t, Pz=vz*t
//   5. 1 D-shell (6 functions) → diagonal with norm, off-diagonal without
//   6. pts > FUNCTIONS_BLOCK_SIZE: multi-block vs CPU
//   7. Mixed S+P vs CPU reference

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC   0

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "test_utils.h"
#include "../../../g2g/common.h"               // FUNCTIONS_BLOCK_SIZE, MAX_CONTRACTIONS
#include "../../../g2g/matrix.h"               // COALESCED_DIMENSION
#include "../../../g2g/scalar_vector_types.h"  // vec_type<T,N>
#include "../../../g2g/cuda/cuda_extra.h"      // index

namespace G2G {
#include "../../../g2g/cuda/gpu_variables.h"   // gpu_atom_positions, gpu_normalization_factor
#include "../../../g2g/cuda/kernels/functions.h"
}

using F4 = G2G::vec_type<float, 4>;
using F2 = G2G::vec_type<float, 2>;

// ---------------------------------------------------------------------------
// CPU reference: evaluate all basis functions for all points.
// Returns function_values[COALESCED_DIM(pts)*func + point].
//
// atom_pos: 3 floats (x,y,z) per atom, atom_pos[3*atom + {0,1,2}]
// pt_xyz:   3 floats (x,y,z) per point, pt_xyz[3*p + {0,1,2}]
// nuc:      nuclear index per function [total_funcs]
// contr:    contraction count per function [total_funcs]
// alphas:   [func * max_c + c] exponent for function func, contraction c
// coeffs:   [func * max_c + c] coefficient
// norm:     normalization factor for D-shell diagonal components
// ---------------------------------------------------------------------------
static std::vector<float> cpu_functions(
    int pts, int n_s, int n_p, int n_d,
    const std::vector<float>& atom_pos,
    const std::vector<float>& pt_xyz,
    const std::vector<unsigned>& nuc,
    const std::vector<unsigned>& contr,
    const std::vector<float>& alphas,
    const std::vector<float>& coeffs,
    int max_c, float norm) {
  int total = n_s + 3 * n_p + 6 * n_d;
  int cdim  = COALESCED_DIMENSION(pts);
  std::vector<float> out(cdim * total, 0.f);

  for (int p = 0; p < pts; p++) {
    float px = pt_xyz[3*p+0], py = pt_xyz[3*p+1], pz = pt_xyz[3*p+2];
    for (int func = 0; func < total; func++) {
      int ni = (int)nuc[func];
      float vx = px - atom_pos[3*ni+0];
      float vy = py - atom_pos[3*ni+1];
      float vz = pz - atom_pos[3*ni+2];
      float dist2 = vx*vx + vy*vy + vz*vz;

      float t = 0.f;
      for (int c = 0; c < (int)contr[func]; c++) {
        float alpha = alphas[func * max_c + c];
        float expon = alpha * dist2;
        if (expon > 70.f) continue;
        t += expf(-expon) * coeffs[func * max_c + c];
      }

      float val;
      if (func < n_s) {
        val = t;
      } else if (func < n_s + 3 * n_p) {
        int p_idx = (func - n_s) % 3;
        float v_comp = (p_idx == 0) ? vx : (p_idx == 1 ? vy : vz);
        val = v_comp * t;
      } else {
        int d_idx = (func - n_s - 3 * n_p) % 6;
        switch (d_idx) {
          case 0: val = norm * vx * vx * t; break;  // XX
          case 1: val = vy * vx * t;         break;  // YX
          case 2: val = norm * vy * vy * t; break;  // YY
          case 3: val = vz * vx * t;         break;  // ZX
          case 4: val = vz * vy * t;         break;  // ZY
          case 5: val = norm * vz * vz * t; break;  // ZZ
          default: val = 0.f;
        }
      }
      out[cdim * func + p] = val;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Upload atom positions and normalization factor to device constant memory.
// atom_pos: 3 floats per atom [n_atoms * 3]
// ---------------------------------------------------------------------------
static void upload_constants(const std::vector<float>& atom_pos, float norm) {
  int n_atoms = (int)atom_pos.size() / 3;
  std::vector<float3> pos3(n_atoms);
  for (int i = 0; i < n_atoms; i++)
    pos3[i] = {atom_pos[3*i+0], atom_pos[3*i+1], atom_pos[3*i+2]};
  CUDA_CHECK(cudaMemcpyToSymbol(G2G::gpu_atom_positions, pos3.data(),
                                n_atoms * sizeof(float3)));
  CUDA_CHECK(cudaMemcpyToSymbol(G2G::gpu_normalization_factor, &norm,
                                sizeof(float)));
}

// ---------------------------------------------------------------------------
// Run gpu_compute_functions<float, false, false>.
// Returns function_values[COALESCED_DIM(pts) * total_funcs].
//
// pt_xyz:  [pts * 3] point coordinates
// nuc:     [total_funcs] nuclear indices
// contr:   [total_funcs] contraction counts
// alphas:  [total_funcs * max_c] exponents
// coeffs:  [total_funcs * max_c] coefficients
// ---------------------------------------------------------------------------
static std::vector<float> run_functions(
    int pts, int n_s, int n_p, int n_d,
    const std::vector<float>& pt_xyz,
    const std::vector<unsigned>& nuc,
    const std::vector<unsigned>& contr,
    const std::vector<float>& alphas,
    const std::vector<float>& coeffs,
    int max_c) {
  int total      = n_s + 3 * n_p + 6 * n_d;
  int cdim_pts   = COALESCED_DIMENSION(pts);
  int cdim_funcs = COALESCED_DIMENSION(total);

  // Build point_positions as F4 (x, y, z, weight=0)
  std::vector<F4> h_pp(pts);
  for (int p = 0; p < pts; p++)
    h_pp[p] = F4(pt_xyz[3*p+0], pt_xyz[3*p+1], pt_xyz[3*p+2], 0.f);

  // Build factor_ac: [COALESCED_DIM(total) * max_c] with .x=alpha, .y=coeff
  std::vector<F2> h_fac((size_t)cdim_funcs * max_c, F2(0.f, 0.f));
  for (int f = 0; f < total; f++)
    for (int c = 0; c < max_c; c++)
      h_fac[(size_t)cdim_funcs * c + f] =
          F2(alphas[f * max_c + c], coeffs[f * max_c + c]);

  // Device allocations
  F4       *d_pp, *d_gv = nullptr;  // gradient_values unused (do_forces=false)
  unsigned *d_nuc, *d_contr;
  F2       *d_fac;
  float    *d_fv;

  CUDA_CHECK(cudaMalloc(&d_pp,    pts * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_nuc,   total * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_contr, total * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_fac,   (size_t)cdim_funcs * max_c * sizeof(F2)));
  CUDA_CHECK(cudaMalloc(&d_fv,    (size_t)cdim_pts * total * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_pp,    h_pp.data(),    pts * sizeof(F4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nuc,   nuc.data(),     total * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_contr, contr.data(),   total * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fac,   h_fac.data(),   cdim_funcs * max_c * sizeof(F2),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_fv, 0, cdim_pts * total * sizeof(float)));

  uint4 functions = {(uint)n_s, (uint)n_p, (uint)n_d, (uint)total};
  dim3 block(FUNCTIONS_BLOCK_SIZE);
  dim3 grid((pts + FUNCTIONS_BLOCK_SIZE - 1) / FUNCTIONS_BLOCK_SIZE);
  G2G::gpu_compute_functions<float, false, false><<<grid, block>>>(
      d_pp, pts, d_contr, d_fac, d_nuc, d_fv, nullptr, nullptr, functions);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_fv((size_t)cdim_pts * total);
  CUDA_CHECK(cudaMemcpy(h_fv.data(), d_fv, cdim_pts * total * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_pp); cudaFree(d_nuc); cudaFree(d_contr);
  cudaFree(d_fac); cudaFree(d_fv);
  return h_fv;
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------
static bool all_near_fv(const std::vector<float>& got,
                         const std::vector<float>& ref,
                         int total_funcs, int pts, float tol = 1e-5f) {
  int cdim = COALESCED_DIMENSION(pts);
  for (int func = 0; func < total_funcs; func++)
    for (int p = 0; p < pts; p++) {
      float g = got[cdim * func + p];
      float r = ref[cdim * func + p];
      if (fabsf(g - r) > tol) {
        printf("    MISMATCH func=%d point=%d: expected %.8g  got %.8g\n",
               func, p, (double)r, (double)g);
        return false;
      }
    }
  return true;
}

// ============================================================================
int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_compute_functions kernel");

  // Normalization factor used in all D-shell tests
  const float norm = sqrtf(3.f);
  const float tol  = 1e-5f;

  // --- 1. S-shell at atom center: phi = coeff ---
  // atom at (0,0,0), point at (0,0,0) → dist=0, t=coeff*exp(0)=coeff
  {
    float coeff = 2.5f, alpha = 1.0f;
    upload_constants({0.f, 0.f, 0.f}, norm);
    auto fv = run_functions(1, 1, 0, 0,
                            {0.f, 0.f, 0.f},   // 1 point
                            {0u}, {1u},         // nuc, contractions
                            {alpha}, {coeff},   // alphas, coeffs
                            1);
    int cdim = COALESCED_DIMENSION(1);
    runner.check(fabsf(fv[cdim*0+0] - coeff) < tol,
                 "S-shell at atom center: phi=coeff");
  }

  // --- 2. S-shell off-center: phi = coeff * exp(-alpha * dist^2) ---
  // atom at (1,0,0), point at (0,0,0) → v=(-1,0,0), dist^2=1
  {
    float alpha = 0.5f, coeff = 3.f;
    float expected = coeff * expf(-alpha * 1.f);
    upload_constants({1.f, 0.f, 0.f}, norm);
    auto fv = run_functions(1, 1, 0, 0,
                            {0.f, 0.f, 0.f},
                            {0u}, {1u},
                            {alpha}, {coeff},
                            1);
    int cdim = COALESCED_DIMENSION(1);
    runner.check(fabsf(fv[cdim*0+0] - expected) < tol,
                 "S-shell off-center: phi=coeff*exp(-a*r^2)");
  }

  // --- 3. S-shell, 2 contractions: phi = sum ---
  // atom at origin, point at (1,0,0) → dist^2=1
  // phi = c1*exp(-a1) + c2*exp(-a2)
  {
    float a1 = 1.f, c1 = 2.f;
    float a2 = 0.5f, c2 = 1.f;
    float expected = c1 * expf(-a1) + c2 * expf(-a2);
    upload_constants({0.f, 0.f, 0.f}, norm);
    auto fv = run_functions(1, 1, 0, 0,
                            {1.f, 0.f, 0.f},
                            {0u}, {2u},
                            {a1, a2}, {c1, c2},
                            2);
    int cdim = COALESCED_DIMENSION(1);
    runner.check(fabsf(fv[cdim*0+0] - expected) < tol,
                 "S-shell 2 contractions");
  }

  // --- 4. P-shell: phi_Px=vx*t, phi_Py=vy*t, phi_Pz=vz*t ---
  // atom at (1,2,3), point at (4,5,6) → v=(3,3,3), dist^2=27
  // t = coeff * exp(-alpha * 27)
  {
    float ax = 1.f, ay = 2.f, az = 3.f;
    float px = 4.f, py = 5.f, pz = 6.f;
    float alpha = 0.1f, coeff = 1.f;
    float vx = px - ax, vy = py - ay, vz = pz - az;
    float dist2 = vx*vx + vy*vy + vz*vz;
    float t = coeff * expf(-alpha * dist2);
    upload_constants({ax, ay, az}, norm);
    // 1 P-shell = 3 functions, functions = {0, 1, 0, 3}
    auto fv = run_functions(1, 0, 1, 0,
                            {px, py, pz},
                            {0u, 0u, 0u}, {1u, 1u, 1u},
                            {alpha, alpha, alpha},
                            {coeff, coeff, coeff},
                            1);
    int cdim = COALESCED_DIMENSION(1);
    bool ok = fabsf(fv[cdim*0+0] - vx*t) < tol &&  // Px
              fabsf(fv[cdim*1+0] - vy*t) < tol &&  // Py
              fabsf(fv[cdim*2+0] - vz*t) < tol;   // Pz
    runner.check(ok, "P-shell: Px=vx*t, Py=vy*t, Pz=vz*t");
  }

  // --- 5. D-shell: all 6 components ---
  // atom at origin, point at (1,2,3) → v=(1,2,3), dist^2=14
  // t = coeff * exp(-alpha * 14)
  {
    float alpha = 0.05f, coeff = 1.f;
    float vx = 1.f, vy = 2.f, vz = 3.f;
    float dist2 = 14.f;
    float t = coeff * expf(-alpha * dist2);
    float expected[6] = {
      norm * vx * vx * t,  // XX
      vy * vx * t,          // YX
      norm * vy * vy * t,  // YY
      vz * vx * t,          // ZX
      vz * vy * t,          // ZY
      norm * vz * vz * t,  // ZZ
    };
    upload_constants({0.f, 0.f, 0.f}, norm);
    // 1 D-shell = 6 functions, functions = {0, 0, 1, 6}
    std::vector<unsigned> nuc6(6, 0u), contr6(6, 1u);
    std::vector<float>    alph6(6, alpha), coef6(6, coeff);
    auto fv = run_functions(1, 0, 0, 1,
                            {vx, vy, vz},
                            nuc6, contr6, alph6, coef6, 1);
    int cdim = COALESCED_DIMENSION(1);
    bool ok = true;
    for (int d = 0; d < 6; d++)
      ok &= fabsf(fv[cdim*d+0] - expected[d]) < tol;
    runner.check(ok, "D-shell: all 6 components with normalization factor");
  }

  // --- 6. pts > FUNCTIONS_BLOCK_SIZE: multi-block vs CPU ---
  {
    int pts = 200, n_s = 3;
    float ax = 0.5f, ay = -0.5f, az = 0.f;
    upload_constants({ax, ay, az}, norm);
    std::vector<float> pt_xyz(pts * 3);
    for (int p = 0; p < pts; p++) {
      pt_xyz[3*p+0] = float(p % 10) * 0.1f;
      pt_xyz[3*p+1] = float(p / 10) * 0.1f;
      pt_xyz[3*p+2] = 0.f;
    }
    std::vector<unsigned> nuc3(n_s, 0u), contr3(n_s, 1u);
    std::vector<float>    alph3 = {0.3f, 0.6f, 1.2f};
    std::vector<float>    coef3 = {1.f,  0.5f, 0.25f};
    auto got = run_functions(pts, n_s, 0, 0, pt_xyz,
                             nuc3, contr3, alph3, coef3, 1);
    auto ref = cpu_functions(pts, n_s, 0, 0,
                             {ax, ay, az}, pt_xyz,
                             nuc3, contr3, alph3, coef3, 1, norm);
    runner.check(all_near_fv(got, ref, n_s, pts, 1e-5f),
                 "pts=200 multi-block vs CPU");
  }

  // --- 7. Mixed S+P vs CPU ---
  {
    int pts = 10, n_s = 2, n_p = 1;  // 2 S + 3 P = 5 functions total
    float ax = 1.f, ay = 0.f, az = 0.f;
    upload_constants({ax, ay, az}, norm);
    std::vector<float> pt_xyz(pts * 3);
    for (int p = 0; p < pts; p++) {
      pt_xyz[3*p+0] = float(p) * 0.2f;
      pt_xyz[3*p+1] = float(p) * 0.1f;
      pt_xyz[3*p+2] = 0.f;
    }
    // 5 functions: S0, S1, Px, Py, Pz – all at atom 0
    std::vector<unsigned> nuc5(5, 0u), contr5(5, 1u);
    std::vector<float>    alph5 = {0.5f, 1.0f, 0.8f, 0.8f, 0.8f};
    std::vector<float>    coef5 = {1.0f, 0.5f, 1.0f, 1.0f, 1.0f};
    auto got = run_functions(pts, n_s, n_p, 0, pt_xyz,
                             nuc5, contr5, alph5, coef5, 1);
    auto ref = cpu_functions(pts, n_s, n_p, 0,
                             {ax, ay, az}, pt_xyz,
                             nuc5, contr5, alph5, coef5, 1, norm);
    runner.check(all_near_fv(got, ref, n_s + 3*n_p, pts, 1e-5f),
                 "mixed S+P vs CPU");
  }

  return runner.summary();
}
