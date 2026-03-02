// CPU conformance tests for GTO basis function evaluation.
//
// Two independent implementations are tested side-by-side:
//   ref_eval_gto_{value,grad,hess_S}() — simple reference in kernels_reference.h
//   G2G::cpu_eval_gto_shell()           — extracted kernel from g2g/cpu/functions.cpp
//
// Each test verifies the reference against a hand-computed value AND then
// verifies the extracted kernel against the reference.  This gives full
// coverage of the real g2g/cpu/functions.cpp computation.
//
// Shell type mapping for cpu_eval_gto_shell():
//   0 = S  (1 function)
//   1 = P  (3 functions: Px, Py, Pz)
//   2 = D  (6 functions: DXX, DXY, DYY, DXZ, DYZ, DZZ)
//
// Shell type mapping for ref_eval_gto_value/grad():
//   0=S, 1=Px, 2=Py, 3=Pz, 4=DXX, 5=DXY, 6=DYY, 7=DXZ, 8=DYZ, 9=DZZ
//
// Test coverage
// -------------
//   1. S-shell value: phi = coeff (at atom center)
//   2. S-shell value off-center: phi = coeff*exp(-alpha*r^2)
//   3. S-shell 2 contractions: phi = sum
//   4. P-shell values: Px=vx*t, Py=vy*t, Pz=vz*t
//   5. D-shell values: all 6 components with normalization
//   6. Exponent cutoff: alpha*r^2 > 70 → contribution skipped
//   7. S-shell gradient: grad = -2*tg*v
//   8. P-shell gradient: product rule d/dk(vx*t)
//   9. D-shell gradient: DXX gradient with normalization
//  10. S-shell Hessian: diagonal and cross second-derivatives

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"
#include "cpu/cpu_kernels.h"   // G2G::cpu_eval_gto_shell (via -I$(G2G_DIR))

// ============================================================================
// Helper to invoke G2G::cpu_eval_gto_shell for a single-contraction S shell
// and return value + optionally gradient and Hessian.
// ============================================================================
struct GtoOut {
  float val[6];
  float gx[6], gy[6], gz[6];
  float hpx[6], hpy[6], hpz[6];
  float hix[6], hiy[6], hiz[6];
};

static GtoOut call_cpu(float vx, float vy, float vz,
                        const std::vector<float>& alphas,
                        const std::vector<float>& coeffs,
                        int shell_type, float norm,
                        bool do_grad, bool do_hess) {
  float dist2 = vx*vx + vy*vy + vz*vz;
  GtoOut o{};
  G2G::cpu_eval_gto_shell<float>(
      vx, vy, vz, dist2,
      alphas.data(), coeffs.data(), (int)alphas.size(),
      shell_type, norm,
      do_grad, do_hess,
      o.val,
      do_grad ? o.gx : nullptr, do_grad ? o.gy : nullptr, do_grad ? o.gz : nullptr,
      do_hess ? o.hpx : nullptr, do_hess ? o.hpy : nullptr, do_hess ? o.hpz : nullptr,
      do_hess ? o.hix : nullptr, do_hess ? o.hiy : nullptr, do_hess ? o.hiz : nullptr);
  return o;
}

// ============================================================================
int main() {
  test_utils::TestRunner runner(
      "GTO basis functions: ref vs cpu_eval_gto_shell");
  const float tol  = 1e-6f;
  const float norm = sqrtf(3.f);

  // --- 1. S-shell at atom center: phi = coeff ---
  {
    float coeff = 2.5f, alpha = 1.0f;
    float dist2 = 0.f;
    float ref = ref_eval_gto_value(0.f, 0.f, 0.f, dist2, {alpha}, {coeff}, 0);
    GtoOut cpu = call_cpu(0.f, 0.f, 0.f, {alpha}, {coeff}, 0, 1.f, false, false);
    bool ok = fabsf(ref - coeff) < tol && fabsf(cpu.val[0] - ref) < tol;
    runner.check(ok, "S-shell at atom center: phi = coeff, ref ≡ cpu");
  }

  // --- 2. S-shell off-center: phi = coeff*exp(-alpha*r^2) ---
  {
    float alpha = 0.5f, coeff = 3.f;
    float vx = -1.f, dist2 = 1.f;
    float expected = coeff * expf(-alpha);
    float ref = ref_eval_gto_value(vx, 0.f, 0.f, dist2, {alpha}, {coeff}, 0);
    GtoOut cpu = call_cpu(vx, 0.f, 0.f, {alpha}, {coeff}, 0, 1.f, false, false);
    bool ok = fabsf(ref - expected) < tol && fabsf(cpu.val[0] - ref) < tol;
    runner.check(ok, "S-shell off-center: phi = coeff*exp(-alpha*r^2), ref ≡ cpu");
  }

  // --- 3. S-shell 2 contractions: phi = sum ---
  {
    std::vector<float> a = {1.f, 0.5f}, c = {2.f, 1.f};
    float vx = 1.f, dist2 = 1.f;
    float expected = c[0]*expf(-a[0]) + c[1]*expf(-a[1]);
    float ref = ref_eval_gto_value(vx, 0.f, 0.f, dist2, a, c, 0);
    GtoOut cpu = call_cpu(vx, 0.f, 0.f, a, c, 0, 1.f, false, false);
    bool ok = fabsf(ref - expected) < tol && fabsf(cpu.val[0] - ref) < tol;
    runner.check(ok, "S-shell 2 contractions: phi = sum, ref ≡ cpu");
  }

  // --- 4. P-shell: Px=vx*t, Py=vy*t, Pz=vz*t, and cpu matches ref ---
  {
    std::vector<float> a = {0.1f}, c = {1.f};
    float vx = 3.f, vy = 3.f, vz = 3.f, dist2 = 27.f;
    float t = c[0] * expf(-a[0] * dist2);
    // ref (shell 1=Px, 2=Py, 3=Pz)
    float ref0 = ref_eval_gto_value(vx,vy,vz,dist2,a,c,1);
    float ref1 = ref_eval_gto_value(vx,vy,vz,dist2,a,c,2);
    float ref2 = ref_eval_gto_value(vx,vy,vz,dist2,a,c,3);
    // cpu (shell_type=1 → 3 functions)
    GtoOut cpu = call_cpu(vx, vy, vz, a, c, 1, 1.f, false, false);
    bool ok = fabsf(ref0 - vx*t) < tol && fabsf(ref1 - vy*t) < tol
           && fabsf(ref2 - vz*t) < tol
           && fabsf(cpu.val[0] - ref0) < tol
           && fabsf(cpu.val[1] - ref1) < tol
           && fabsf(cpu.val[2] - ref2) < tol;
    runner.check(ok, "P-shell: Px/Py/Pz values, ref ≡ cpu");
  }

  // --- 5. D-shell: all 6 components with normalization, cpu matches ref ---
  {
    std::vector<float> a = {0.05f}, c = {1.f};
    float vx = 1.f, vy = 2.f, vz = 3.f, dist2 = 14.f;
    float t = c[0] * expf(-a[0] * dist2);
    float expected[6] = {
      norm*vx*vx*t, vy*vx*t, norm*vy*vy*t,
      vz*vx*t,      vz*vy*t, norm*vz*vz*t
    };
    // ref (shell types 4..9)
    bool ok = true;
    for (int d = 0; d < 6; ++d)
      ok &= fabsf(ref_eval_gto_value(vx,vy,vz,dist2,a,c,4+d,norm) - expected[d]) < tol;
    // cpu (shell_type=2 → 6 functions)
    GtoOut cpu = call_cpu(vx, vy, vz, a, c, 2, norm, false, false);
    for (int d = 0; d < 6; ++d)
      ok &= fabsf(cpu.val[d] - expected[d]) < tol;
    runner.check(ok, "D-shell: all 6 components with normalization, ref ≡ cpu");
  }

  // --- 6. Exponent cutoff: alpha*r^2 > 70 → contribution = 0 ---
  {
    float vx = 1.f;
    GtoOut skip = call_cpu(vx, 0.f, 0.f, {100.f}, {5.f}, 0, 1.f, false, false);
    GtoOut ok_  = call_cpu(vx, 0.f, 0.f, {1.f},   {5.f}, 0, 1.f, false, false);
    bool ok = fabsf(skip.val[0]) < tol
           && fabsf(ok_.val[0] - 5.f*expf(-1.f)) < tol;
    runner.check(ok, "exponent cutoff: alpha*r^2>70 → contribution=0, cpu");
  }

  // --- 7. S-shell gradient: grad = -2*tg*v, ref ≡ cpu ---
  {
    std::vector<float> a = {0.5f}, c = {1.f};
    float vx = 1.f, vy = 0.f, vz = 0.f, dist2 = 1.f;
    float tg = a[0] * c[0] * expf(-a[0] * dist2);
    float exp_gx = -2.f * tg * vx;
    RefVec3 refg = ref_eval_gto_grad(vx, vy, vz, dist2, a, c, 0);
    GtoOut  cpu  = call_cpu(vx, vy, vz, a, c, 0, 1.f, true, false);
    bool ok = fabsf(refg.x - exp_gx) < tol && fabsf(refg.y) < tol
           && fabsf(refg.z) < tol
           && fabsf(cpu.gx[0] - refg.x) < tol
           && fabsf(cpu.gy[0] - refg.y) < tol
           && fabsf(cpu.gz[0] - refg.z) < tol;
    runner.check(ok, "S-shell gradient: grad = -2*tg*v, ref ≡ cpu");
  }

  // --- 8. P-shell (Px) gradient: product rule, ref ≡ cpu ---
  {
    std::vector<float> a = {0.1f}, c = {1.f};
    float vx = 1.f, vy = 2.f, vz = 0.f, dist2 = 5.f;
    float t  = c[0] * expf(-a[0] * dist2);
    float tg = a[0] * t;
    float exp_gx = t - 2.f*tg*vx*vx;
    float exp_gy =   - 2.f*tg*vy*vx;
    RefVec3 refg = ref_eval_gto_grad(vx, vy, vz, dist2, a, c, 1);  // Px
    GtoOut  cpu  = call_cpu(vx, vy, vz, a, c, 1, 1.f, true, false);
    bool ok = fabsf(refg.x - exp_gx) < tol && fabsf(refg.y - exp_gy) < tol
           && fabsf(refg.z) < tol
           && fabsf(cpu.gx[0] - refg.x) < tol
           && fabsf(cpu.gy[0] - refg.y) < tol
           && fabsf(cpu.gz[0] - refg.z) < tol;
    runner.check(ok, "Px-shell gradient: product rule, ref ≡ cpu");
  }

  // --- 9. D-shell (DXX) gradient: cpu matches ref ---
  {
    std::vector<float> a = {0.1f}, c = {1.f};
    float vx = 1.f, vy = 0.f, vz = 0.f, dist2 = 1.f;
    float t  = c[0] * expf(-a[0] * dist2);
    float tg = a[0] * t;
    float exp_gx = norm * (2.f*vx*t - 2.f*tg*vx*vx*vx);
    RefVec3 refg = ref_eval_gto_grad(vx, vy, vz, dist2, a, c, 4, norm);  // DXX
    GtoOut  cpu  = call_cpu(vx, vy, vz, a, c, 2, norm, true, false);
    bool ok = fabsf(refg.x - exp_gx) < tol && fabsf(refg.y) < tol
           && fabsf(refg.z) < tol
           && fabsf(cpu.gx[0] - refg.x) < tol
           && fabsf(cpu.gy[0] - refg.y) < tol
           && fabsf(cpu.gz[0] - refg.z) < tol;
    runner.check(ok, "DXX-shell gradient: norm*(2vx*t - 2tg*vx^3), ref ≡ cpu");
  }

  // --- 10. S-shell Hessian: diagonal and cross second-derivatives, ref ≡ cpu ---
  {
    std::vector<float> a = {0.5f}, c = {1.f};
    float vx = 1.f, vy = 0.f, vz = 0.f, dist2 = 1.f;
    float t0 = c[0] * expf(-a[0] * dist2);
    float tg = a[0] * t0, th = a[0] * a[0] * t0;
    float exp_px = vx*vx*4.f*th - 2.f*tg;
    float exp_py =               - 2.f*tg;  // vy=0
    RefHess refh = ref_eval_gto_hess_S(vx, vy, vz, dist2, a, c);
    GtoOut  cpu  = call_cpu(vx, vy, vz, a, c, 0, 1.f, true, true);
    bool ok = fabsf(refh.px - exp_px) < tol && fabsf(refh.py - exp_py) < tol
           && fabsf(refh.ix) < tol && fabsf(refh.iy) < tol && fabsf(refh.iz) < tol
           && fabsf(cpu.hpx[0] - refh.px) < tol
           && fabsf(cpu.hpy[0] - refh.py) < tol
           && fabsf(cpu.hpz[0] - refh.pz) < tol
           && fabsf(cpu.hix[0] - refh.ix) < tol
           && fabsf(cpu.hiy[0] - refh.iy) < tol
           && fabsf(cpu.hiz[0] - refh.iz) < tol;
    runner.check(ok, "S-shell Hessian: diagonal and cross, ref ≡ cpu");
  }

  return runner.summary();
}
