// CPU conformance tests for GTO basis function evaluation.
//
// Tests the same mathematical formula implemented in:
//   g2g/cuda/kernels/functions.h  (GPU version)
//   g2g/cpu/functions.cpp         (CPU version, inside PointGroupCPU)
//
// For a basis function centered on atom A = (Ax, Ay, Az):
//   v = r - R_A  (vector from atom to grid point)
//   t = sum_c coeff_c * exp(-alpha_c * |v|^2)  (contracted radial part)
//
//   S-shell: phi = t
//   P-shell: phi_Px = vx*t,  phi_Py = vy*t,  phi_Pz = vz*t
//   D-shell: phi_XX = norm*vx^2*t,  phi_XY = vy*vx*t,  phi_YY = norm*vy^2*t
//            phi_XZ = vz*vx*t,      phi_YZ = vz*vy*t,  phi_ZZ = norm*vz^2*t
//
// The D-shell diagonal components (XX, YY, ZZ) are scaled by the
// normalization factor (typ. sqrt(3)), matching the GPU kernel.
//
// Test coverage
// -------------
//   1. S-shell: point at atom center → phi = coeff
//   2. S-shell: point off-center → phi = coeff * exp(-alpha*r^2)
//   3. S-shell: 2 contractions → phi = sum
//   4. P-shell: verify Px, Py, Pz angular components
//   5. D-shell: verify all 6 components
//   6. Exponent cutoff: alpha*r^2 > 70 → contribution is skipped

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"

// ---------------------------------------------------------------------------
// Core GTO radial part.
// Matches the cutoff condition (expon > 70) used by both GPU and CPU kernels.
// ---------------------------------------------------------------------------
static float gto_radial(float dist2,
                         const std::vector<float>& alphas,
                         const std::vector<float>& coeffs) {
  float t = 0.f;
  for (int c = 0; c < (int)alphas.size(); c++) {
    float expon = alphas[c] * dist2;
    if (expon > 70.f) continue;
    t += expf(-expon) * coeffs[c];
  }
  return t;
}

// ---------------------------------------------------------------------------
// Evaluate one GTO at a single point.
// shell_type: 0=S, 1=Px, 2=Py, 3=Pz, 4=DXX, 5=DXY, 6=DYY, 7=DXZ, 8=DYZ, 9=DZZ
// norm: normalization factor for D-shell diagonal components
// ---------------------------------------------------------------------------
static float eval_gto(float px, float py, float pz,
                       float ax, float ay, float az,
                       const std::vector<float>& alphas,
                       const std::vector<float>& coeffs,
                       int shell_type, float norm = 1.f) {
  float vx = px - ax, vy = py - ay, vz = pz - az;
  float dist2 = vx*vx + vy*vy + vz*vz;
  float t = gto_radial(dist2, alphas, coeffs);

  switch (shell_type) {
    case 0: return t;                   // S
    case 1: return vx * t;              // Px
    case 2: return vy * t;              // Py
    case 3: return vz * t;              // Pz
    case 4: return norm * vx * vx * t; // DXX
    case 5: return vy * vx * t;        // DXY (YX in kernel)
    case 6: return norm * vy * vy * t; // DYY
    case 7: return vz * vx * t;        // DXZ (ZX in kernel)
    case 8: return vz * vy * t;        // DYZ (ZY in kernel)
    case 9: return norm * vz * vz * t; // DZZ
    default: return 0.f;
  }
}

// ============================================================================
int main() {
  test_utils::TestRunner runner("GTO basis functions (CPU algorithm)");
  const float tol  = 1e-6f;
  const float norm = sqrtf(3.f);  // typical D-shell normalization in LIO

  // --- 1. S-shell: point at atom center → phi = coeff ---
  {
    float coeff = 2.5f, alpha = 1.0f;
    float phi = eval_gto(0.f, 0.f, 0.f,   // point at origin
                         0.f, 0.f, 0.f,   // atom at origin
                         {alpha}, {coeff}, 0, norm);
    runner.check(fabsf(phi - coeff) < tol,
                 "S-shell at atom center: phi = coeff");
  }

  // --- 2. S-shell: point off-center → phi = coeff * exp(-alpha * r^2) ---
  // atom at (1,0,0), point at (0,0,0), r^2=1
  {
    float alpha = 0.5f, coeff = 3.f;
    float expected = coeff * expf(-alpha * 1.f);
    float phi = eval_gto(0.f, 0.f, 0.f,
                         1.f, 0.f, 0.f,
                         {alpha}, {coeff}, 0, norm);
    runner.check(fabsf(phi - expected) < tol,
                 "S-shell off-center: phi = coeff*exp(-alpha*r^2)");
  }

  // --- 3. S-shell, 2 contractions: phi = sum of contributions ---
  // atom at (0,0,0), point at (1,0,0), r^2=1
  {
    float a1 = 1.f, c1 = 2.f;
    float a2 = 0.5f, c2 = 1.f;
    float expected = c1 * expf(-a1) + c2 * expf(-a2);
    float phi = eval_gto(1.f, 0.f, 0.f,
                         0.f, 0.f, 0.f,
                         {a1, a2}, {c1, c2}, 0, norm);
    runner.check(fabsf(phi - expected) < tol,
                 "S-shell 2 contractions: phi = sum");
  }

  // --- 4. P-shell: verify all 3 angular components ---
  // atom at (1,2,3), point at (4,5,6): v=(3,3,3), r^2=27
  {
    float alpha = 0.1f, coeff = 1.f;
    float vx = 3.f, vy = 3.f, vz = 3.f;
    float r2 = 27.f;
    float t = coeff * expf(-alpha * r2);
    float px_v = eval_gto(4.f, 5.f, 6.f, 1.f, 2.f, 3.f,
                          {alpha}, {coeff}, 1, norm);
    float py_v = eval_gto(4.f, 5.f, 6.f, 1.f, 2.f, 3.f,
                          {alpha}, {coeff}, 2, norm);
    float pz_v = eval_gto(4.f, 5.f, 6.f, 1.f, 2.f, 3.f,
                          {alpha}, {coeff}, 3, norm);
    bool ok = fabsf(px_v - vx*t) < tol &&
              fabsf(py_v - vy*t) < tol &&
              fabsf(pz_v - vz*t) < tol;
    runner.check(ok, "P-shell: Px=vx*t, Py=vy*t, Pz=vz*t");
  }

  // --- 5. D-shell: all 6 components ---
  // atom at (0,0,0), point at (1,2,3): v=(1,2,3), r^2=14
  {
    float alpha = 0.05f, coeff = 1.f;
    float vx = 1.f, vy = 2.f, vz = 3.f;
    float r2 = 14.f;
    float t = coeff * expf(-alpha * r2);
    float expected[6] = {
      norm * vx * vx * t,  // XX
      vy * vx * t,          // XY (= YX in kernel ordering)
      norm * vy * vy * t,  // YY
      vz * vx * t,          // XZ (= ZX in kernel ordering)
      vz * vy * t,          // YZ (= ZY in kernel ordering)
      norm * vz * vz * t,  // ZZ
    };
    int shell_types[6] = {4, 5, 6, 7, 8, 9};
    const char* names[6] = {"DXX", "DXY", "DYY", "DXZ", "DYZ", "DZZ"};
    bool ok = true;
    for (int d = 0; d < 6; d++) {
      float phi = eval_gto(vx, vy, vz, 0.f, 0.f, 0.f,
                           {alpha}, {coeff}, shell_types[d], norm);
      if (fabsf(phi - expected[d]) > tol) {
        printf("    MISMATCH %s: expected %.8g  got %.8g\n",
               names[d], (double)expected[d], (double)phi);
        ok = false;
      }
    }
    runner.check(ok, "D-shell: all 6 components with normalization");
  }

  // --- 6. Exponent cutoff: alpha*r^2 > 70 → contribution skipped ---
  // For alpha=100 and r^2=1 → expon=100 > 70 → t=0 (skipped)
  // For alpha=1 and r^2=1 → expon=1 ≤ 70 → t=coeff*exp(-1)
  {
    float phi_skip = eval_gto(1.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                              {100.f}, {5.f}, 0, norm);
    float phi_ok   = eval_gto(1.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                              {1.f},   {5.f}, 0, norm);
    bool ok = fabsf(phi_skip - 0.f) < tol &&
              fabsf(phi_ok - 5.f * expf(-1.f)) < tol;
    runner.check(ok, "exponent cutoff: alpha*r^2>70 → contribution=0");
  }

  return runner.summary();
}
