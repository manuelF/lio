// CPU conformance tests for the force integration algorithm.
//
// Tests ref_cpu_forces<RefF4> from kernels_reference.h.  This is the same
// algorithm that gpu_compute_forces (force.h) validates against, tested here
// in isolation to confirm the mathematical formula is correct.
//
// Formula: forces[a] = sum_{p} density_deriv[COALESCED_DIM(pts)*a+p] * factor[p]
// The derivs array uses COALESCED_DIMENSION(pts) stride between atom rows.
//
// Test coverage
// -------------
//   1. Zero factors → zero forces
//   2. Single atom, single point: force = deriv * factor
//   3. n_atoms=2, pts=3: verify against hand-computed sum
//   4. Scale linearity: factor × 2 → force × 2
//   5. pts=300 (many points): verify against explicit loop

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"

using F4 = RefF4;

static bool near4(F4 a, F4 b, float tol = 1e-5f) {
  return fabsf(a.x-b.x)<=tol && fabsf(a.y-b.y)<=tol &&
         fabsf(a.z-b.z)<=tol && fabsf(a.w-b.w)<=tol;
}

// Build COALESCED_DIMENSION(pts)-padded derivs from row-major vals[atom][point].
static std::vector<F4> make_derivs(int n_atoms, int pts,
                                    const std::vector<F4>& vals) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4> out(cdim * n_atoms);
  for (int a = 0; a < n_atoms; ++a)
    for (int p = 0; p < pts; ++p)
      out[cdim*a+p] = vals[a*pts+p];
  return out;
}

// ============================================================================
int main() {
  test_utils::TestRunner runner("ref_cpu_forces algorithm");

  // --- 1. Zero factors → zero forces ---
  {
    int na = 3, pts = 5;
    std::vector<float> factors(pts, 0.f);
    std::vector<F4>    vals(na * pts, F4(1.f, 2.f, 3.f, 0.f));
    auto derivs = make_derivs(na, pts, vals);
    auto forces = ref_cpu_forces<F4>(na, pts, factors, derivs);
    bool ok = true;
    for (auto& f : forces) ok &= near4(f, F4(0.f, 0.f, 0.f, 0.f));
    runner.check(ok, "zero factors → zero forces");
  }

  // --- 2. Single atom, single point: force = deriv * factor ---
  // derivs[0] = (1,2,3,0), factor=2  → force = (2,4,6,0)
  {
    int na = 1, pts = 1;
    std::vector<float> factors = {2.f};
    auto derivs = make_derivs(1, 1, {F4(1.f, 2.f, 3.f, 0.f)});
    auto forces = ref_cpu_forces<F4>(na, pts, factors, derivs);
    runner.check(near4(forces[0], F4(2.f, 4.f, 6.f, 0.f)),
                 "single atom single point: force = deriv * factor");
  }

  // --- 3. n_atoms=2, pts=3: hand-computed sum ---
  // atom0 derivs: (1,0,0,0),(0,1,0,0),(0,0,1,0)  factors: 1,2,3
  // force[0] = (1*1+0*2+0*3, 0*1+1*2+0*3, 0*1+0*2+1*3, 0) = (1,2,3,0)
  // atom1 derivs: (2,0,0,0),(0,2,0,0),(0,0,2,0)
  // force[1] = (2*1, 2*2, 2*3, 0) = (2,4,6,0)
  {
    int na = 2, pts = 3;
    std::vector<float> factors = {1.f, 2.f, 3.f};
    std::vector<F4> vals = {
      F4(1.f,0.f,0.f,0.f), F4(0.f,1.f,0.f,0.f), F4(0.f,0.f,1.f,0.f),  // atom0
      F4(2.f,0.f,0.f,0.f), F4(0.f,2.f,0.f,0.f), F4(0.f,0.f,2.f,0.f),  // atom1
    };
    auto derivs = make_derivs(na, pts, vals);
    auto forces = ref_cpu_forces<F4>(na, pts, factors, derivs);
    bool ok = near4(forces[0], F4(1.f, 2.f, 3.f, 0.f)) &&
              near4(forces[1], F4(2.f, 4.f, 6.f, 0.f));
    runner.check(ok, "n_atoms=2 pts=3 hand-computed sum");
  }

  // --- 4. Scale linearity: factor × 2 → force × 2 ---
  {
    int na = 2, pts = 5;
    std::vector<float> f1(pts), f2(pts);
    std::vector<F4> vals(na*pts);
    for (int p = 0; p < pts; ++p) { f1[p] = float(p+1)*0.3f; f2[p] = 2.f*f1[p]; }
    for (int a = 0; a < na; ++a)
      for (int p = 0; p < pts; ++p)
        vals[a*pts+p] = F4(float(a+p+1)*0.1f, float(p)*0.2f, 0.f, 0.f);
    auto derivs = make_derivs(na, pts, vals);
    auto forces1 = ref_cpu_forces<F4>(na, pts, f1, derivs);
    auto forces2 = ref_cpu_forces<F4>(na, pts, f2, derivs);
    bool ok = true;
    for (int a = 0; a < na; ++a)
      ok &= near4(forces2[a], F4(2.f*forces1[a].x, 2.f*forces1[a].y,
                                  2.f*forces1[a].z, 2.f*forces1[a].w));
    runner.check(ok, "scale factor×2 → force×2 (linear)");
  }

  // --- 5. pts=300 (many points): multi-row stride vs explicit loop ---
  {
    int na = 2, pts = 300;
    int cdim = COALESCED_DIMENSION(pts);
    std::vector<float> factors(pts);
    std::vector<F4> vals(na*pts);
    for (int p = 0; p < pts; ++p) factors[p] = sinf(float(p)*0.05f);
    for (int a = 0; a < na; ++a)
      for (int p = 0; p < pts; ++p)
        vals[a*pts+p] = F4(cosf(float(p+a)), sinf(float(p+a)), 0.f, 0.f);
    auto derivs = make_derivs(na, pts, vals);
    auto forces = ref_cpu_forces<F4>(na, pts, factors, derivs);

    // Independent verification: explicit per-atom explicit sum
    bool ok = true;
    for (int a = 0; a < na; ++a) {
      float ex = 0.f, ey = 0.f;
      for (int p = 0; p < pts; ++p) {
        ex += derivs[cdim*a+p].x * factors[p];
        ey += derivs[cdim*a+p].y * factors[p];
      }
      ok &= fabsf(forces[a].x - ex) < 3e-4f && fabsf(forces[a].y - ey) < 3e-4f;
    }
    runner.check(ok, "pts=300 multi-stride vs explicit loop");
  }

  return runner.summary();
}
