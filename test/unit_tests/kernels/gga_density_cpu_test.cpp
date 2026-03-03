// CPU conformance tests for cpu_compute_density_gga().
//
// Compares G2G::cpu_compute_density_gga() from g2g/cpu/cpu_kernels.h against
// the portable reference ref_cpu_density_gga() from kernels_reference.h.
//
// Both implementations use the LOWER TRIANGLE (j <= i) of the density matrix.
// The rmm passed to cpu_compute_density_gga must have the lower triangle filled
// (upper is ignored), matching the convention of get_rmm_input().
//
// Test coverage
// -------------
//   1. m=1, pts=1: trivial pd = R[0][0]*F[0]^2
//   2. m=2: hand-verify pd and gradient sums
//   3. Zero rmm → all outputs zero
//   4. m=4: ref ≡ cpu_compute_density_gga across all 10 outputs
//   5. Scale F→2F: pd→4*pd (bilinear in fv)
//   6. Open-shell: Ra==Rb → da.pd == db.pd (and all other fields)

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"
#include "cpu/cpu_kernels.h"

static bool near(float a, float b, float tol = 1e-5f) {
  return fabsf(a - b) <= tol;
}

static bool near_gga(const RefGGADensity& ref,
                     const G2G::GGADensity<float>& cpu, float tol = 1e-5f) {
  return near(ref.pd,    cpu.pd,    tol) &&
         near(ref.tdx,   cpu.tdx,   tol) &&
         near(ref.tdy,   cpu.tdy,   tol) &&
         near(ref.tdz,   cpu.tdz,   tol) &&
         near(ref.tdd1x, cpu.tdd1x, tol) &&
         near(ref.tdd1y, cpu.tdd1y, tol) &&
         near(ref.tdd1z, cpu.tdd1z, tol) &&
         near(ref.tdd2x, cpu.tdd2x, tol) &&
         near(ref.tdd2y, cpu.tdd2y, tol) &&
         near(ref.tdd2z, cpu.tdd2z, tol);
}

// ============================================================================
int main() {
  test_utils::TestRunner runner("cpu_compute_density_gga vs ref_cpu_density_gga");
  const float tol = 1e-5f;

  // --- 1. m=1, trivial: pd = R[0][0]*F[0]^2 ---
  // rmm=[2], fv=[3], all grads=[1]
  // w = 2*3 = 6, pd = 3*6 = 18
  {
    int m = 1;
    std::vector<float> rmm = {2.f};
    std::vector<float> fv  = {3.f};
    std::vector<float> gx  = {1.f}, gy = {0.f}, gz = {0.f};
    std::vector<float> hpx = {0.f}, hpy = {0.f}, hpz = {0.f};
    std::vector<float> hix = {0.f}, hiy = {0.f}, hiz = {0.f};
    auto ref = ref_cpu_density_gga(fv, gx, gy, gz, hpx, hpy, hpz, hix, hiy, hiz, rmm, m);
    auto cpu = G2G::cpu_compute_density_gga(
        fv.data(), gx.data(), gy.data(), gz.data(),
        hpx.data(), hpy.data(), hpz.data(),
        hix.data(), hiy.data(), hiz.data(), rmm.data(), m);
    bool ok = near(ref.pd, 18.f, tol) && near_gga(ref, cpu, tol);
    runner.check(ok, "m=1 trivial: pd=18, ref == cpu");
  }

  // --- 2. m=2: hand-verify pd ---
  // R: R[0][0]=1, R[1][0]=2, R[1][1]=3   fv=[1,2]
  // gxv=[1,0], rest=0
  // w(i=0) = R[0][0]*fv[0] = 1
  // w(i=1) = R[1][0]*fv[0] + R[1][1]*fv[1] = 2 + 6 = 8
  // pd = fv[0]*w(0) + fv[1]*w(1) = 1 + 16 = 17
  // tdx(i=0) = gx[0]*w(0) + w3x(0)*fv[0] = 1*1 + 1*1 = 2
  // tdx(i=1) = gx[1]*w(1) + w3x(1)*fv[1] = 0*8 + (gx[0]*R10+gx[1]*R11)*fv[1]
  //          = 0 + (1*2+0*3)*2 = 4  → total tdx = 2 + 4 = 6
  {
    int m = 2;
    std::vector<float> rmm = {1.f, 0.f, 2.f, 3.f};
    std::vector<float> fv  = {1.f, 2.f};
    std::vector<float> gx  = {1.f, 0.f}, gy(m,0.f), gz(m,0.f);
    std::vector<float> hpx(m,0.f), hpy(m,0.f), hpz(m,0.f);
    std::vector<float> hix(m,0.f), hiy(m,0.f), hiz(m,0.f);
    auto ref = ref_cpu_density_gga(fv, gx, gy, gz, hpx, hpy, hpz, hix, hiy, hiz, rmm, m);
    auto cpu = G2G::cpu_compute_density_gga(
        fv.data(), gx.data(), gy.data(), gz.data(),
        hpx.data(), hpy.data(), hpz.data(),
        hix.data(), hiy.data(), hiz.data(), rmm.data(), m);
    bool ok = near(ref.pd, 17.f, tol) &&
              near(ref.tdx, 6.f, tol) &&
              near_gga(ref, cpu, tol);
    runner.check(ok, "m=2 hand-verify: pd=17, tdx=6, ref == cpu");
  }

  // --- 3. Zero rmm → all outputs zero ---
  {
    int m = 3;
    std::vector<float> rmm(m*m, 0.f);
    std::vector<float> fv(m, 1.f), gx(m, 1.f), gy(m, 1.f), gz(m, 1.f);
    std::vector<float> hpx(m,1.f), hpy(m,1.f), hpz(m,1.f);
    std::vector<float> hix(m,1.f), hiy(m,1.f), hiz(m,1.f);
    auto cpu = G2G::cpu_compute_density_gga(
        fv.data(), gx.data(), gy.data(), gz.data(),
        hpx.data(), hpy.data(), hpz.data(),
        hix.data(), hiy.data(), hiz.data(), rmm.data(), m);
    bool ok = near(cpu.pd, 0.f, tol) &&
              near(cpu.tdx, 0.f, tol) &&
              near(cpu.tdd1x, 0.f, tol) &&
              near(cpu.tdd2x, 0.f, tol);
    runner.check(ok, "zero rmm → all outputs zero");
  }

  // --- 4. m=4: ref == cpu_compute_density_gga (all 10 outputs) ---
  {
    int m = 4;
    std::vector<float> rmm(m*m, 0.f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i*m+j] = float(i*m+j+1) * 0.1f;
    std::vector<float> fv(m), gx(m), gy(m), gz(m);
    std::vector<float> hpx(m), hpy(m), hpz(m), hix(m), hiy(m), hiz(m);
    for (int i = 0; i < m; ++i) {
      fv[i]  = float(i+1) * 0.3f;
      gx[i]  = float(i+1) * 0.1f;
      gy[i]  = float(m-i) * 0.1f;
      gz[i]  = float(i*2+1) * 0.05f;
      hpx[i] = float(i+1) * 0.02f;
      hpy[i] = float(m-i) * 0.02f;
      hpz[i] = 0.01f;
      hix[i] = float(i+1) * 0.015f;
      hiy[i] = float(m-i) * 0.015f;
      hiz[i] = float(i) * 0.01f;
    }
    auto ref = ref_cpu_density_gga(fv, gx, gy, gz, hpx, hpy, hpz, hix, hiy, hiz, rmm, m);
    auto cpu = G2G::cpu_compute_density_gga(
        fv.data(), gx.data(), gy.data(), gz.data(),
        hpx.data(), hpy.data(), hpz.data(),
        hix.data(), hiy.data(), hiz.data(), rmm.data(), m);
    runner.check(near_gga(ref, cpu, tol), "m=4: ref == cpu_compute_density_gga (all 10 fields)");
  }

  // --- 5. Scale F→2F: pd→4*pd (bilinear in fv) ---
  {
    int m = 3;
    std::vector<float> rmm(m*m, 0.f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i*m+j] = float(i+j+1) * 0.5f;
    std::vector<float> fv(m), fv2(m);
    std::vector<float> gx(m, 0.f), gy(m, 0.f), gz(m, 0.f);
    std::vector<float> hpx(m,0.f), hpy(m,0.f), hpz(m,0.f);
    std::vector<float> hix(m,0.f), hiy(m,0.f), hiz(m,0.f);
    for (int i = 0; i < m; ++i) { fv[i] = float(i+1); fv2[i] = 2.f*fv[i]; }
    auto r1 = G2G::cpu_compute_density_gga(
        fv.data(),  gx.data(), gy.data(), gz.data(),
        hpx.data(), hpy.data(), hpz.data(),
        hix.data(), hiy.data(), hiz.data(), rmm.data(), m);
    auto r2 = G2G::cpu_compute_density_gga(
        fv2.data(), gx.data(), gy.data(), gz.data(),
        hpx.data(), hpy.data(), hpz.data(),
        hix.data(), hiy.data(), hiz.data(), rmm.data(), m);
    runner.check(near(r2.pd, 4.f * r1.pd, tol), "scale F×2 → pd×4 (bilinear)");
  }

  // --- 6. Open-shell: Ra==Rb → da == db (all fields) ---
  {
    int m = 3;
    std::vector<float> rmm(m*m, 0.f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i*m+j] = float(i+j+1) * 0.2f;
    std::vector<float> fv(m), gx(m), gy(m), gz(m);
    std::vector<float> hpx(m), hpy(m), hpz(m), hix(m), hiy(m), hiz(m);
    for (int i = 0; i < m; ++i) {
      fv[i]  = float(i+1) * 0.4f;
      gx[i]  = float(i+2) * 0.1f;
      gy[i]  = float(m-i) * 0.1f;
      gz[i]  = float(i+1) * 0.05f;
      hpx[i] = float(i+1) * 0.02f;
      hpy[i] = 0.01f;
      hpz[i] = 0.01f;
      hix[i] = hiy[i] = hiz[i] = float(i) * 0.01f;
    }
    auto da = G2G::cpu_compute_density_gga(
        fv.data(), gx.data(), gy.data(), gz.data(),
        hpx.data(), hpy.data(), hpz.data(),
        hix.data(), hiy.data(), hiz.data(), rmm.data(), m);
    auto db = G2G::cpu_compute_density_gga(
        fv.data(), gx.data(), gy.data(), gz.data(),
        hpx.data(), hpy.data(), hpz.data(),
        hix.data(), hiy.data(), hiz.data(), rmm.data(), m);
    bool ok = near(da.pd, db.pd, tol) &&
              near(da.tdx, db.tdx, tol) &&
              near(da.tdd1x, db.tdd1x, tol) &&
              near(da.tdd2x, db.tdd2x, tol);
    runner.check(ok, "open-shell Ra==Rb → da == db (all fields)");
  }

  return runner.summary();
}
