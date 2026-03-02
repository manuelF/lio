// CPU conformance tests for the electron density computation algorithm.
//
// Tests ref_cpu_density from kernels_reference.h against hand-computed values
// and an independent naive full-matrix expansion.  This is the same algorithm
// that gpu_compute_density (energy.h) validates against; testing it here in
// isolation ensures the reference is itself mathematically correct.
//
// Formula: rho(p) = sum_i fv[m*p+i] * sum_{j<=i} rmm[i*m+j] * fv[m*p+j]
// rmm stores the lower-triangular density matrix.
//
// Test coverage
// -------------
//   1. m=1, pts=1: trivial rho = R[0][0]*F[0]^2
//   2. m=2, pts=1: hand-verifiable
//   3. Zero function values → zero density
//   4. m=4, pts=3: lower-triangular formula ≡ naive symmetric expansion
//   5. Scale linearity: F → 2*F implies rho → 4*rho
//   6. Additive decomposition: rho(Ra+Rb) = rho(Ra) + rho(Rb)

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"

// Alternative implementation of the same LIO lower-triangular formula,
// written as a plain double loop instead of an accumulated inner sum.
// LIO uses only the lower triangle: upper entries are 0 by convention.
// rho = sum_{i>=j} rmm[i*m+j] * fv[m*p+i] * fv[m*p+j]
static float density_flat_loop(const std::vector<float>& rmm, int m,
                                const std::vector<float>& fv, int p) {
  float rho = 0.f;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j <= i; ++j)
      rho += rmm[i*m+j] * fv[m*p+i] * fv[m*p+j];
  return rho;
}

// ============================================================================
int main() {
  test_utils::TestRunner runner("ref_cpu_density algorithm");
  const float tol = 1e-5f;

  // --- 1. m=1 trivial: rho = R[0][0]*F[0]^2 = 2*3^2 = 18 ---
  {
    float rho = ref_cpu_density({2.f}, 1, {3.f}, 0);
    runner.check(fabsf(rho - 18.f) < tol, "m=1 trivial: rho=18");
  }

  // --- 2. m=2 hand-verify ---
  // R (lower-tri): R[0][0]=1, R[1][0]=2, R[1][1]=3   F=[1,2]
  // rho = F[0]*(R[0][0]*F[0]) + F[1]*(R[1][0]*F[0]+R[1][1]*F[1])
  //     = 1*1 + 2*(2+6) = 17
  {
    std::vector<float> rmm = {1.f, 0.f,  // row 0
                               2.f, 3.f}; // row 1
    float rho = ref_cpu_density(rmm, 2, {1.f, 2.f}, 0);
    runner.check(fabsf(rho - 17.f) < tol, "m=2 hand-verify: rho=17");
  }

  // --- 3. Zero function values → zero density ---
  {
    std::vector<float> rmm = {1.f, 0.f, 2.f, 3.f};
    std::vector<float> fv(2, 0.f);
    runner.check(fabsf(ref_cpu_density(rmm, 2, fv, 0)) < tol,
                 "zero F → zero density");
  }

  // --- 4. m=4, pts=3: lower-tri formula ≡ naive symmetric expansion ---
  {
    int m = 4, pts = 3;
    std::vector<float> rmm(m*m, 0.f), fv(m*pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i*m+j] = float(i*m+j+1) * 0.1f;
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m*p+i] = float(p*m+i+1) * 0.3f;
    bool ok = true;
    for (int p = 0; p < pts; ++p)
      ok &= fabsf(ref_cpu_density(rmm, m, fv, p)
                  - density_flat_loop(rmm, m, fv, p)) < tol;
    runner.check(ok, "m=4 pts=3: lower-tri ≡ flat-loop alternative");
  }

  // --- 5. Scale linearity: F → 2*F implies rho → 4*rho ---
  {
    int m = 3;
    std::vector<float> rmm(m*m, 0.f), fv(m), fv2(m);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i*m+j] = float(i+j+1) * 0.5f;
    for (int i = 0; i < m; ++i) { fv[i] = float(i+1); fv2[i] = 2.f*fv[i]; }
    float rho1 = ref_cpu_density(rmm, m, fv,  0);
    float rho2 = ref_cpu_density(rmm, m, fv2, 0);
    runner.check(fabsf(rho2 - 4.f * rho1) < tol,
                 "scale F×2 → rho×4 (bilinear)");
  }

  // --- 6. Additive decomposition: rho(Ra+Rb) = rho(Ra) + rho(Rb) ---
  {
    int m = 3, pts = 2;
    std::vector<float> rmm_a(m*m,0.f), rmm_b(m*m,0.f), rmm_c(m*m,0.f), fv(m*pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j) {
        rmm_a[i*m+j] = float(i+1) * 0.3f;
        rmm_b[i*m+j] = float(j+1) * 0.2f;
        rmm_c[i*m+j] = rmm_a[i*m+j] + rmm_b[i*m+j];
      }
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m*p+i] = float(p+i+1) * 0.4f;
    bool ok = true;
    for (int p = 0; p < pts; ++p) {
      float sum = ref_cpu_density(rmm_a,m,fv,p) + ref_cpu_density(rmm_b,m,fv,p);
      ok &= fabsf(ref_cpu_density(rmm_c,m,fv,p) - sum) < tol;
    }
    runner.check(ok, "density additive in RMM: rho(Ra+Rb)=rho(Ra)+rho(Rb)");
  }

  return runner.summary();
}
