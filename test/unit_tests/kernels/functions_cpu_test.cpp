// CPU conformance tests for GTO basis function evaluation.
//
// Tests the same mathematical formula implemented in:
//   g2g/cuda/kernels/functions.h  (GPU version)
//   g2g/cpu/functions.cpp         (CPU version, inside PointGroupCPU)
//
// Uses ref_eval_gto_value / ref_eval_gto_grad / ref_eval_gto_hess_S from
// kernels_reference.h, which are the shared reference implementations also
// used to validate the CUDA functions_test.cu kernel outputs.
//
// Shell ordering:
//   0=S, 1=Px, 2=Py, 3=Pz,
//   4=DXX, 5=DXY(=YX), 6=DYY, 7=DXZ(=ZX), 8=DYZ(=ZY), 9=DZZ
// D-shell diagonal (XX, YY, ZZ) are scaled by normalization_factor = sqrt(3).
//
// Test coverage: values (1-6), gradients (7-9), Hessian (10)
//   1. S-shell: point at atom center → phi = coeff
//   2. S-shell: point off-center → phi = coeff * exp(-alpha*r^2)
//   3. S-shell: 2 contractions → phi = sum
//   4. P-shell: verify Px, Py, Pz angular components
//   5. D-shell: verify all 6 components with normalization
//   6. Exponent cutoff: alpha*r^2 > 70 → contribution is skipped
//   7. S-shell gradient: grad(t) = -2*tg*(vx, vy, vz)
//   8. P-shell gradient: verify Px gradient via product rule
//   9. D-shell gradient: verify DXX gradient with normalization
//  10. S-shell Hessian: verify diagonal and cross second-derivatives

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"

// ============================================================================
int main() {
  test_utils::TestRunner runner("GTO basis functions (CPU algorithm)");
  const float tol  = 1e-6f;
  const float norm = sqrtf(3.f);  // typical D-shell normalization in LIO

  // Helper: evaluate GTO at a point given as offset from atom.
  // Computes dist2 from vx,vy,vz internally.
  auto gto_val = [&](float vx, float vy, float vz,
                     const std::vector<float>& a, const std::vector<float>& c,
                     int shell, float n = 1.f) -> float {
    float dist2 = vx*vx + vy*vy + vz*vz;
    return ref_eval_gto_value(vx, vy, vz, dist2, a, c, shell, n);
  };
  auto gto_grad = [&](float vx, float vy, float vz,
                      const std::vector<float>& a, const std::vector<float>& c,
                      int shell, float n = 1.f) -> RefVec3 {
    float dist2 = vx*vx + vy*vy + vz*vz;
    return ref_eval_gto_grad(vx, vy, vz, dist2, a, c, shell, n);
  };

  // --- 1. S-shell at atom center → phi = coeff ---
  {
    float coeff = 2.5f, alpha = 1.0f;
    float phi = gto_val(0.f, 0.f, 0.f, {alpha}, {coeff}, 0, norm);
    runner.check(fabsf(phi - coeff) < tol,
                 "S-shell at atom center: phi = coeff");
  }

  // --- 2. S-shell off-center → phi = coeff * exp(-alpha*r^2) ---
  // v=(−1,0,0), r^2=1
  {
    float alpha = 0.5f, coeff = 3.f;
    float expected = coeff * expf(-alpha * 1.f);
    float phi = gto_val(-1.f, 0.f, 0.f, {alpha}, {coeff}, 0, norm);
    runner.check(fabsf(phi - expected) < tol,
                 "S-shell off-center: phi = coeff*exp(-alpha*r^2)");
  }

  // --- 3. S-shell, 2 contractions: phi = sum of contributions ---
  // v=(1,0,0), r^2=1
  {
    float a1 = 1.f, c1 = 2.f, a2 = 0.5f, c2 = 1.f;
    float expected = c1*expf(-a1) + c2*expf(-a2);
    float phi = gto_val(1.f, 0.f, 0.f, {a1, a2}, {c1, c2}, 0, norm);
    runner.check(fabsf(phi - expected) < tol,
                 "S-shell 2 contractions: phi = sum");
  }

  // --- 4. P-shell: verify all 3 angular components ---
  // v=(3,3,3), r^2=27, alpha=0.1, coeff=1
  {
    float alpha = 0.1f, coeff = 1.f;
    float vx = 3.f, vy = 3.f, vz = 3.f, r2 = 27.f;
    float t = coeff * expf(-alpha * r2);
    bool ok = fabsf(gto_val(vx,vy,vz,{alpha},{coeff},1,norm) - vx*t) < tol &&
              fabsf(gto_val(vx,vy,vz,{alpha},{coeff},2,norm) - vy*t) < tol &&
              fabsf(gto_val(vx,vy,vz,{alpha},{coeff},3,norm) - vz*t) < tol;
    runner.check(ok, "P-shell: Px=vx*t, Py=vy*t, Pz=vz*t");
  }

  // --- 5. D-shell: all 6 components with normalization ---
  // v=(1,2,3), r^2=14, alpha=0.05, coeff=1
  {
    float alpha = 0.05f, coeff = 1.f;
    float vx = 1.f, vy = 2.f, vz = 3.f, r2 = 14.f;
    float t = coeff * expf(-alpha * r2);
    float expected[6] = {
      norm * vx * vx * t,  // DXX (shell 4)
      vy * vx * t,          // DXY (shell 5)
      norm * vy * vy * t,  // DYY (shell 6)
      vz * vx * t,          // DXZ (shell 7)
      vz * vy * t,          // DYZ (shell 8)
      norm * vz * vz * t,  // DZZ (shell 9)
    };
    bool ok = true;
    for (int d = 0; d < 6; d++)
      ok &= fabsf(gto_val(vx,vy,vz,{alpha},{coeff},4+d,norm) - expected[d]) < tol;
    runner.check(ok, "D-shell: all 6 components with normalization");
  }

  // --- 6. Exponent cutoff: alpha*r^2 > 70 → contribution skipped ---
  {
    float phi_skip = gto_val(1.f,0.f,0.f, {100.f},{5.f}, 0, norm);
    float phi_ok   = gto_val(1.f,0.f,0.f, {1.f},  {5.f}, 0, norm);
    bool ok = fabsf(phi_skip) < tol &&
              fabsf(phi_ok - 5.f * expf(-1.f)) < tol;
    runner.check(ok, "exponent cutoff: alpha*r^2>70 → contribution=0");
  }

  // --- 7. S-shell gradient: grad(t) = -2*tg*(vx, vy, vz) ---
  // v=(1,0,0), r^2=1, alpha=0.5, coeff=1
  // tg = alpha*exp(-alpha) = 0.5*exp(-0.5)
  // grad = (-2*0.5*exp(-0.5), 0, 0) = (-exp(-0.5), 0, 0)
  {
    float alpha = 0.5f, coeff = 1.f;
    float tg = alpha * coeff * expf(-alpha * 1.f);
    float expected_gx = -2.f * tg;
    RefVec3 g = gto_grad(1.f, 0.f, 0.f, {alpha}, {coeff}, 0, norm);
    bool ok = fabsf(g.x - expected_gx) < tol &&
              fabsf(g.y) < tol &&
              fabsf(g.z) < tol;
    runner.check(ok, "S-shell gradient: grad = -2*tg*v");
  }

  // --- 8. P-shell (Px) gradient via product rule ---
  // v=(1,2,0), r^2=5, alpha=0.1, coeff=1
  // t  = exp(-0.5),  tg = 0.1*exp(-0.5)
  // grad Px: gx = t - 2*tg*vx^2,  gy = -2*tg*vy*vx,  gz = 0
  {
    float alpha = 0.1f, coeff = 1.f;
    float vx = 1.f, vy = 2.f, vz = 0.f, r2 = 5.f;
    float t  = coeff * expf(-alpha * r2);
    float tg = alpha * t;
    float expected_gx = t - 2.f * tg * vx * vx;
    float expected_gy =   - 2.f * tg * vy * vx;
    float expected_gz = 0.f;
    RefVec3 g = gto_grad(vx, vy, vz, {alpha}, {coeff}, 1, norm);
    bool ok = fabsf(g.x - expected_gx) < tol &&
              fabsf(g.y - expected_gy) < tol &&
              fabsf(g.z - expected_gz) < tol;
    runner.check(ok, "Px-shell gradient: product rule d/dk(vx*t)");
  }

  // --- 9. D-shell (DXX) gradient with normalization ---
  // v=(1,0,0), r^2=1, alpha=0.1, coeff=1, norm=sqrt(3)
  // t  = exp(-0.1),  tg = 0.1*exp(-0.1)
  // grad DXX: gx = norm*(2*vx*t - 2*tg*vx^3), gy = gz = 0
  {
    float alpha = 0.1f, coeff = 1.f;
    float vx = 1.f, vy = 0.f, vz = 0.f, r2 = 1.f;
    float t  = coeff * expf(-alpha * r2);
    float tg = alpha * t;
    float expected_gx = norm * (2.f * vx * t - 2.f * tg * vx * vx * vx);
    float expected_gy = 0.f;
    float expected_gz = 0.f;
    RefVec3 g = gto_grad(vx, vy, vz, {alpha}, {coeff}, 4, norm);
    bool ok = fabsf(g.x - expected_gx) < tol &&
              fabsf(g.y - expected_gy) < tol &&
              fabsf(g.z - expected_gz) < tol;
    runner.check(ok, "DXX-shell gradient: norm*(2*vx*t - 2*tg*vx^3)");
  }

  // --- 10. S-shell Hessian ---
  // v=(1,0,0), r^2=1, alpha=0.5, coeff=1
  // tg = 0.5*exp(-0.5),  th = 0.25*exp(-0.5)
  // hPX = vx^2*4*th - 2*tg = 4*0.25*e^{-0.5} - 2*0.5*e^{-0.5} = 0
  // hPY = vy^2*4*th - 2*tg = 0 - e^{-0.5} = -e^{-0.5}
  // hPZ = same as hPY
  // hIX = vx*vy*4*th = 0,  hIY = vx*vz*4*th = 0,  hIZ = vy*vz*4*th = 0
  {
    float alpha = 0.5f, coeff = 1.f;
    float r2 = 1.f;
    float t0 = coeff * expf(-alpha * r2);
    float tg = alpha * t0;
    float th = alpha * alpha * t0;
    float expected_px = 1.f * 1.f * 4.f * th - 2.f * tg;  // = 0
    float expected_py = 0.f * 0.f * 4.f * th - 2.f * tg;  // = -e^{-0.5}
    float expected_pz = expected_py;
    float dist2 = r2;
    RefHess h = ref_eval_gto_hess_S(1.f, 0.f, 0.f, dist2, {alpha}, {coeff});
    bool ok = fabsf(h.px - expected_px) < tol &&
              fabsf(h.py - expected_py) < tol &&
              fabsf(h.pz - expected_pz) < tol &&
              fabsf(h.ix) < tol &&
              fabsf(h.iy) < tol &&
              fabsf(h.iz) < tol;
    runner.check(ok, "S-shell Hessian: diagonal and cross second-derivatives");
  }

  return runner.summary();
}
