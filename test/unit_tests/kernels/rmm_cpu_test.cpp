// CPU conformance tests for cpu_update_rmm().
//
// Compares G2G::cpu_update_rmm() from g2g/cpu/cpu_kernels.h against
// the portable reference ref_cpu_update_rmm() from kernels_reference.h.
//
// Formula:  result = sum_{p=0}^{npoints-1} fv_row[p] * fv_col[p] * factors[p]
//
// This is the inner dot product used in the RMM update section of
// solve_closed() and solve_opened().
//
// Test coverage
// -------------
//   1. npoints=1: trivial result = fvr[0]*fvc[0]*factor[0]
//   2. npoints=3: hand-verify
//   3. Zero factors → zero result
//   4. npoints=5: ref ≡ cpu_update_rmm
//   5. Scale factors×2 → result×2 (linearity in factors)
//   6. Open-shell: same fv, different factors → different results

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"
#include "cpu/cpu_kernels.h"

// ============================================================================
int main() {
  test_utils::TestRunner runner("cpu_update_rmm vs ref_cpu_update_rmm");
  const float tol = 1e-5f;

  // --- 1. npoints=1: trivial result = 2*3*4 = 24 ---
  {
    std::vector<float> fvr = {2.f}, fvc = {3.f}, fac = {4.f};
    float ref = ref_cpu_update_rmm(fvr, fvc, fac, 1);
    float cpu = G2G::cpu_update_rmm(fvr.data(), fvc.data(), fac.data(), 1);
    bool ok = fabsf(ref - 24.f) < tol && fabsf(cpu - ref) < tol;
    runner.check(ok, "npoints=1 trivial: result=24, ref == cpu");
  }

  // --- 2. npoints=3: hand-verify ---
  // (1*1*1) + (2*2*2) + (3*3*3) = 1 + 8 + 27 = 36
  {
    std::vector<float> fvr = {1.f, 2.f, 3.f};
    std::vector<float> fvc = {1.f, 2.f, 3.f};
    std::vector<float> fac = {1.f, 2.f, 3.f};
    float ref = ref_cpu_update_rmm(fvr, fvc, fac, 3);
    float cpu = G2G::cpu_update_rmm(fvr.data(), fvc.data(), fac.data(), 3);
    bool ok = fabsf(ref - 36.f) < tol && fabsf(cpu - ref) < tol;
    runner.check(ok, "npoints=3 hand-verify: result=36, ref == cpu");
  }

  // --- 3. Zero factors → zero result ---
  {
    std::vector<float> fvr = {1.f, 2.f, 3.f};
    std::vector<float> fvc = {4.f, 5.f, 6.f};
    std::vector<float> fac(3, 0.f);
    float cpu = G2G::cpu_update_rmm(fvr.data(), fvc.data(), fac.data(), 3);
    runner.check(fabsf(cpu) < tol, "zero factors → zero result");
  }

  // --- 4. npoints=5: ref ≡ cpu_update_rmm ---
  {
    int n = 5;
    std::vector<float> fvr(n), fvc(n), fac(n);
    for (int p = 0; p < n; ++p) {
      fvr[p] = float(p+1) * 0.3f;
      fvc[p] = float(n-p) * 0.4f;
      fac[p] = float(p+1) * 0.2f;
    }
    float ref = ref_cpu_update_rmm(fvr, fvc, fac, n);
    float cpu = G2G::cpu_update_rmm(fvr.data(), fvc.data(), fac.data(), n);
    runner.check(fabsf(cpu - ref) < tol, "npoints=5: ref ≡ cpu_update_rmm");
  }

  // --- 5. Scale factors×2 → result×2 (linearity in factors) ---
  {
    int n = 4;
    std::vector<float> fvr(n), fvc(n), fac(n), fac2(n);
    for (int p = 0; p < n; ++p) {
      fvr[p] = float(p+1) * 0.5f;
      fvc[p] = float(n-p) * 0.3f;
      fac[p]  = float(p+1) * 0.1f;
      fac2[p] = 2.f * fac[p];
    }
    float r1 = G2G::cpu_update_rmm(fvr.data(), fvc.data(), fac.data(),  n);
    float r2 = G2G::cpu_update_rmm(fvr.data(), fvc.data(), fac2.data(), n);
    runner.check(fabsf(r2 - 2.f * r1) < tol, "scale factors×2 → result×2");
  }

  // --- 6. Open-shell: same fv, different factors → different results ---
  {
    int n = 4;
    std::vector<float> fvr(n), fvc(n), fac_a(n), fac_b(n);
    for (int p = 0; p < n; ++p) {
      fvr[p]   = float(p+1) * 0.4f;
      fvc[p]   = float(n-p) * 0.4f;
      fac_a[p] = float(p+1) * 0.3f;
      fac_b[p] = float(n-p) * 0.2f;
    }
    float res_a = G2G::cpu_update_rmm(fvr.data(), fvc.data(), fac_a.data(), n);
    float res_b = G2G::cpu_update_rmm(fvr.data(), fvc.data(), fac_b.data(), n);
    float ref_a = ref_cpu_update_rmm(fvr, fvc, fac_a, n);
    float ref_b = ref_cpu_update_rmm(fvr, fvc, fac_b, n);
    bool ok = fabsf(res_a - ref_a) < tol &&
              fabsf(res_b - ref_b) < tol &&
              fabsf(res_a - res_b) > tol;  // actually different
    runner.check(ok, "open-shell: different factors → different results, both match ref");
  }

  return runner.summary();
}
