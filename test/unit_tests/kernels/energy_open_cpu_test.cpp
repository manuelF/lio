// CPU conformance tests for the open-shell electron density algorithm.
//
// Open-shell systems have two separate density matrices (alpha and beta spin).
// Each spin density is computed independently using ref_cpu_density:
//   rho_alpha(p) = sum_i fv[m*p+i] * sum_{j<=i} Ra[i][j] * fv[m*p+j]
//   rho_beta(p)  = sum_i fv[m*p+i] * sum_{j<=i} Rb[i][j] * fv[m*p+j]
//
// Both use the same basis function values fv but different RMMs.
// This mirrors gpu_compute_density_opened (energy_open.h) which takes two
// texture objects tex_a and tex_b.
//
// Test coverage
// -------------
//   1. Ra==Rb → rho_alpha == rho_beta (same matrix, same density)
//   2. Zero RMM → zero density
//   3. Ra = 2*Rb → rho_alpha = 2*rho_beta (linearity in RMM)
//   4. m=4, pts=3: alpha and beta independently verified

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"

// ============================================================================
int main() {
  test_utils::TestRunner runner("ref_cpu_density open-shell algorithm");
  const float tol = 1e-5f;

  // --- 1. Ra==Rb → rho_alpha == rho_beta ---
  {
    int m = 4, pts = 3;
    std::vector<float> r(m*m, 0.f), fv(m*pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        r[i*m+j] = sinf(float(i+j)*0.3f);
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m*p+i] = cosf(float(p*m+i)*0.1f);
    bool ok = true;
    for (int p = 0; p < pts; ++p)
      ok &= fabsf(ref_cpu_density(r, m, fv, p) - ref_cpu_density(r, m, fv, p)) < tol;
    runner.check(ok, "Ra==Rb → rho_a == rho_b");
  }

  // --- 2. Zero RMM → zero density ---
  {
    int m = 3, pts = 2;
    std::vector<float> r(m*m, 0.f), fv(m*pts);
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m*p+i] = float(p*m+i+1)*0.5f;
    bool ok = true;
    for (int p = 0; p < pts; ++p)
      ok &= fabsf(ref_cpu_density(r, m, fv, p)) < tol;
    runner.check(ok, "zero RMM → zero density");
  }

  // --- 3. Ra = 2*Rb → rho_alpha = 2*rho_beta (linearity in RMM) ---
  {
    int m = 3, pts = 2;
    std::vector<float> rb(m*m, 0.f), ra(m*m, 0.f), fv(m*pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j) {
        rb[i*m+j] = float(i+j+1)*0.2f;
        ra[i*m+j] = 2.f * rb[i*m+j];
      }
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m*p+i] = float(p+i+1)*0.3f;
    bool ok = true;
    for (int p = 0; p < pts; ++p)
      ok &= fabsf(ref_cpu_density(ra, m, fv, p)
                  - 2.f * ref_cpu_density(rb, m, fv, p)) < tol;
    runner.check(ok, "Ra=2*Rb → rho_alpha=2*rho_beta");
  }

  // --- 4. m=4, pts=3: alpha and beta values independently verified ---
  // Ra: R[0][0]=1, R[1][0]=2, R[1][1]=3  F_0=[1,1,0,0]
  // rho_a(p=0) = F0*(R00*F0)+F1*(R10*F0+R11*F1) = 1*1 + 1*(2+3) = 6
  // Rb: R[0][0]=2 only (rest 0)   F_0=[1,1,0,0]
  // rho_b(p=0) = 1*(2*1) + 1*(0+0) = 2
  {
    int m = 4, pts = 3;
    std::vector<float> ra(m*m, 0.f), rb(m*m, 0.f), fv(m*pts, 0.f);
    ra[0*m+0] = 1.f; ra[1*m+0] = 2.f; ra[1*m+1] = 3.f;
    rb[0*m+0] = 2.f;
    fv[0] = 1.f; fv[1] = 1.f;  // p=0: F=[1,1,0,0]
    fv[4] = 0.5f; fv[5] = 0.5f; fv[6] = 1.f;  // p=1
    fv[8] = 1.f; fv[9] = 0.f; fv[10] = 0.f; fv[11] = 2.f;  // p=2
    float rho_a0 = ref_cpu_density(ra, m, fv, 0);
    float rho_b0 = ref_cpu_density(rb, m, fv, 0);
    bool ok = fabsf(rho_a0 - 6.f) < tol && fabsf(rho_b0 - 2.f) < tol;
    runner.check(ok, "m=4 pts=3: alpha and beta independently verified");
  }

  return runner.summary();
}
