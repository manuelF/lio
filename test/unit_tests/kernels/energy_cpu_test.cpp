// CPU conformance tests for the electron density computation algorithm.
//
// Two independent implementations are tested side-by-side:
//   ref_cpu_density()                — simple reference in kernels_reference.h
//   G2G::cpu_compute_density_lda()  — extracted kernel from g2g/cpu/iteration.cpp
//
// The two implementations use complementary triangle conventions:
//   ref: lower-triangle  rmm[i*m+j] for j <= i (upper triangle = 0)
//   cpu: upper-triangle  rmm[i*m+j] for j >= i (uses fully symmetric matrix)
// Given a symmetrized RMM both produce identical densities.
//
// Formula: rho(p) = sum_i fv[p*m+i] * sum_{j<=i} rmm[i*m+j] * fv[p*m+j]
// rmm stores the lower-triangular density matrix (ref convention).
//
// Test coverage
// -------------
//   1. m=1, pts=1: trivial rho = R[0][0]*F[0]^2 = 18
//   2. m=2, pts=1: hand-verifiable rho = 17
//   3. Zero function values → zero density
//   4. m=4, pts=3: ref ≡ cpu_compute_density_lda (both implementations)
//   5. Scale linearity: F → 2*F implies rho → 4*rho
//   6. Additive decomposition: rho(Ra+Rb) = rho(Ra) + rho(Rb)

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"
#include "cpu/cpu_kernels.h"   // G2G::cpu_compute_density_lda (via -I$(G2G_DIR))

// Symmetrize a lower-triangular RMM for use with cpu_compute_density_lda.
// Lower-tri convention: rmm_lower[i*m+j] is non-zero only for j <= i.
// Symmetric result: rmm_sym[i*m+j] = rmm_sym[j*m+i] = rmm_lower[max(i,j)*m+min(i,j)]
static std::vector<float> symmetrize(const std::vector<float>& lower, int m) {
  std::vector<float> sym(m * m, 0.f);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j <= i; ++j)
      sym[i*m+j] = sym[j*m+i] = lower[i*m+j];
  return sym;
}

// ============================================================================
int main() {
  test_utils::TestRunner runner("ref_cpu_density vs cpu_compute_density_lda");
  const float tol = 1e-5f;

  // --- 1. m=1 trivial: rho = R[0][0]*F[0]^2 = 2*3^2 = 18 ---
  {
    std::vector<float> rmm_lower = {2.f};
    std::vector<float> fv = {3.f};
    float ref = ref_cpu_density(rmm_lower, 1, fv, 0);
    auto  sym = symmetrize(rmm_lower, 1);
    float cpu = G2G::cpu_compute_density_lda(fv.data(), sym.data(), 1);
    bool ok = fabsf(ref - 18.f) < tol && fabsf(cpu - ref) < tol;
    runner.check(ok, "m=1 trivial: rho=18, ref ≡ cpu");
  }

  // --- 2. m=2 hand-verify: rho = 17 ---
  // R: R[0][0]=1, R[1][0]=2, R[1][1]=3   F=[1,2]
  // rho = F[0]*(R[0][0]*F[0]) + F[1]*(R[1][0]*F[0]+R[1][1]*F[1]) = 1 + 2*(2+6) = 17
  {
    std::vector<float> rmm_lower = {1.f, 0.f, 2.f, 3.f};
    std::vector<float> fv = {1.f, 2.f};
    float ref = ref_cpu_density(rmm_lower, 2, fv, 0);
    auto  sym = symmetrize(rmm_lower, 2);
    float cpu = G2G::cpu_compute_density_lda(fv.data(), sym.data(), 2);
    bool ok = fabsf(ref - 17.f) < tol && fabsf(cpu - ref) < tol;
    runner.check(ok, "m=2 hand-verify: rho=17, ref ≡ cpu");
  }

  // --- 3. Zero function values → zero density ---
  {
    std::vector<float> rmm_lower = {1.f, 0.f, 2.f, 3.f};
    std::vector<float> fv(2, 0.f);
    float ref = ref_cpu_density(rmm_lower, 2, fv, 0);
    auto  sym = symmetrize(rmm_lower, 2);
    float cpu = G2G::cpu_compute_density_lda(fv.data(), sym.data(), 2);
    bool ok = fabsf(ref) < tol && fabsf(cpu) < tol;
    runner.check(ok, "zero F → zero density, ref ≡ cpu");
  }

  // --- 4. m=4, pts=3: ref ≡ cpu_compute_density_lda for all points ---
  {
    int m = 4, pts = 3;
    std::vector<float> rmm_lower(m*m, 0.f), fv(m*pts);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm_lower[i*m+j] = float(i*m+j+1) * 0.1f;
    for (int p = 0; p < pts; ++p)
      for (int i = 0; i < m; ++i)
        fv[m*p+i] = float(p*m+i+1) * 0.3f;
    auto sym = symmetrize(rmm_lower, m);
    bool ok = true;
    for (int p = 0; p < pts; ++p) {
      float ref = ref_cpu_density(rmm_lower, m, fv, p);
      float cpu = G2G::cpu_compute_density_lda(&fv[m*p], sym.data(), m);
      ok &= fabsf(cpu - ref) < tol;
    }
    runner.check(ok, "m=4 pts=3: ref ≡ cpu_compute_density_lda");
  }

  // --- 5. Scale linearity: F → 2*F implies rho → 4*rho ---
  {
    int m = 3;
    std::vector<float> rmm_lower(m*m, 0.f), fv(m), fv2(m);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm_lower[i*m+j] = float(i+j+1) * 0.5f;
    for (int i = 0; i < m; ++i) { fv[i] = float(i+1); fv2[i] = 2.f*fv[i]; }
    float rho1 = ref_cpu_density(rmm_lower, m, fv,  0);
    float rho2 = ref_cpu_density(rmm_lower, m, fv2, 0);
    bool ok = fabsf(rho2 - 4.f * rho1) < tol;
    runner.check(ok, "scale F×2 → rho×4 (bilinear)");
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
