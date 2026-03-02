// CPU conformance tests for the density derivative computation algorithm.
//
// Tests ref_cpu_density_derivs<RefF4> from kernels_reference.h.  This is the
// same algorithm that gpu_compute_density_derivs (energy_derivs.h) validates
// against, tested here in isolation.
//
// Formula: deriv[COALESCED_DIM(pts)*nuc[i]+p] -= grad_phi_i(p) * w_i
//   where w_i = sum_k R[k][i] * fv[cdim*k+p] * (k==i ? 2 : 1)
//   and R[k][i] = rmm[k*m+i]
//
// fv layout: fv[COALESCED_DIM(pts)*func + point]
// gv layout: gv[COALESCED_DIM(pts)*func + point]
//
// Test coverage
// -------------
//   1. m=1, pts=1: trivial w = 2*R[0][0]*F[0]
//   2. m=2, pts=1, 2 atoms: hand-verify
//   3. Zero gradient → zero derivative
//   4. m=4, pts=3: lower-tri formula ≡ full-matrix naive expansion
//   5. Open-shell: Ra==Rb → deriv_a == deriv_b

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

// Build COALESCED_DIMENSION layout: input vals[func*pts+point] → out[cdim*func+point]
static std::vector<float> make_fv(int m, int pts, const std::vector<float>& vals) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<float> out(cdim * m, 0.f);
  for (int f = 0; f < m; ++f)
    for (int p = 0; p < pts; ++p)
      out[cdim*f+p] = vals[f*pts+p];
  return out;
}
static std::vector<F4> make_gv(int m, int pts, const std::vector<F4>& vals) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4> out(cdim * m);
  for (int f = 0; f < m; ++f)
    for (int p = 0; p < pts; ++p)
      out[cdim*f+p] = vals[f*pts+p];
  return out;
}

// Alternative implementation of the same LIO lower-triangular formula,
// written as explicit loops instead of the accumulated inner sum.
// LIO uses only the lower triangle: upper entries are 0 by convention.
// w_i = 2*R[i][i]*fv[i,p]  +  sum_{k>i} R[k][i]*fv[k,p]
// (R[k][i] = rmm[k*m+i] for k >= i, 0 otherwise — never symmetrize)
static std::vector<F4> derivs_flat_loop(
    const std::vector<float>& lower_rmm, int m,
    const std::vector<float>& fv, const std::vector<F4>& gv,
    const std::vector<unsigned>& nuc, int nuc_count, int pts) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4> deriv(cdim * nuc_count);
  for (auto& d : deriv) d = F4(0.f,0.f,0.f,0.f);

  for (int p = 0; p < pts; ++p) {
    for (int i = 0; i < m; ++i) {
      float w = 2.f * lower_rmm[i*m+i] * fv[cdim*i+p];
      for (int k = i+1; k < m; ++k)
        w += lower_rmm[k*m+i] * fv[cdim*k+p];
      int ni = (int)nuc[i];
      deriv[cdim*ni+p].x -= gv[cdim*i+p].x * w;
      deriv[cdim*ni+p].y -= gv[cdim*i+p].y * w;
      deriv[cdim*ni+p].z -= gv[cdim*i+p].z * w;
      deriv[cdim*ni+p].w -= gv[cdim*i+p].w * w;
    }
  }
  return deriv;
}

// ============================================================================
int main() {
  test_utils::TestRunner runner("ref_cpu_density_derivs algorithm");
  const float tol = 1e-5f;

  // --- 1. m=1, pts=1: trivial ---
  // R=[2], fv=[3], gv=[(1,0,0,0)], nuc=[0]
  // w = R[0][0]*3*2 = 12, deriv[atom0] = -(1,0,0,0)*12 = (-12,0,0,0)
  {
    auto fv = make_fv(1, 1, {3.f});
    auto gv = make_gv(1, 1, {F4(1.f,0.f,0.f,0.f)});
    auto deriv = ref_cpu_density_derivs<F4>({2.f}, 1, fv, gv, {0u}, 1, 1);
    int cdim = COALESCED_DIMENSION(1);
    runner.check(near4(deriv[cdim*0+0], F4(-12.f,0.f,0.f,0.f), tol),
                 "m=1 pts=1 trivial: deriv[0]=(-12,0,0,0)");
  }

  // --- 2. m=2, pts=1, 2 atoms: hand-verify ---
  // R: R[0][0]=1, R[1][0]=2, R[1][1]=3    fv=[1,2]
  // gv0=(1,0,0,0), gv1=(0,1,0,0), nuc=[0,1]
  // i=0: w = R[0][0]*1*2 + R[1][0]*2*1 = 2+4=6   → deriv[atom0] -= (1,0,0)*6
  // i=1: w = R[0][1]*1*1 + R[1][1]*2*2 = 0+12=12 → deriv[atom1] -= (0,1,0)*12
  {
    std::vector<float> rmm = {1.f, 0.f, 2.f, 3.f};
    auto fv = make_fv(2, 1, {1.f, 2.f});
    auto gv = make_gv(2, 1, {F4(1.f,0.f,0.f,0.f), F4(0.f,1.f,0.f,0.f)});
    auto deriv = ref_cpu_density_derivs<F4>(rmm, 2, fv, gv, {0u,1u}, 2, 1);
    int cdim = COALESCED_DIMENSION(1);
    bool ok = near4(deriv[cdim*0+0], F4(-6.f,  0.f, 0.f, 0.f), tol) &&
              near4(deriv[cdim*1+0], F4( 0.f, -12.f, 0.f, 0.f), tol);
    runner.check(ok, "m=2 pts=1 two atoms: hand-verify");
  }

  // --- 3. Zero gradient → zero derivatives ---
  {
    std::vector<float> rmm = {1.f, 0.f, 2.f, 3.f};
    auto fv = make_fv(2, 3, {1.f,2.f,3.f,  4.f,5.f,6.f});
    auto gv = make_gv(2, 3, std::vector<F4>(6, F4(0.f,0.f,0.f,0.f)));
    auto deriv = ref_cpu_density_derivs<F4>(rmm, 2, fv, gv, {0u,1u}, 2, 3);
    bool ok = true;
    int cdim = COALESCED_DIMENSION(3);
    for (int na = 0; na < 2; ++na)
      for (int p = 0; p < 3; ++p)
        ok &= near4(deriv[cdim*na+p], F4(0.f,0.f,0.f,0.f), tol);
    runner.check(ok, "zero gradient → zero derivatives");
  }

  // --- 4. m=4, pts=3: lower-tri formula ≡ naive full-symmetric expansion ---
  {
    int m = 4, pts = 3, nuc_count = 2;
    std::vector<float> rmm(m*m, 0.f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i*m+j] = float(i*m+j+1) * 0.1f;
    std::vector<float> fv_raw(m*pts);
    std::vector<F4>    gv_raw(m*pts);
    std::vector<unsigned> nuc = {0u,0u,1u,1u};
    for (int f = 0; f < m; ++f)
      for (int p = 0; p < pts; ++p) {
        fv_raw[f*pts+p] = float(f*pts+p+1)*0.3f;
        gv_raw[f*pts+p] = F4(float(f+p)*0.1f, float(p-f)*0.1f, float(f)*0.05f, 0.f);
      }
    auto fv   = make_fv(m, pts, fv_raw);
    auto gv   = make_gv(m, pts, gv_raw);
    auto ref1 = ref_cpu_density_derivs<F4>(rmm, m, fv, gv, nuc, nuc_count, pts);
    auto ref2 = derivs_flat_loop(rmm, m, fv, gv, nuc, nuc_count, pts);
    bool ok = true;
    int cdim = COALESCED_DIMENSION(pts);
    for (int na = 0; na < nuc_count; ++na)
      for (int p = 0; p < pts; ++p)
        ok &= near4(ref1[cdim*na+p], ref2[cdim*na+p], tol);
    runner.check(ok, "m=4 pts=3: lower-tri ≡ flat-loop alternative");
  }

  // --- 5. Ra==Rb → deriv_a == deriv_b (open-shell symmetry) ---
  {
    int m = 4, pts = 5, nuc_count = 2;
    std::vector<float> rmm(m*m, 0.f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm[i*m+j] = float(i+j+1)*0.15f;
    std::vector<float> fv_raw(m*pts);
    std::vector<F4>    gv_raw(m*pts);
    std::vector<unsigned> nuc = {0u,0u,1u,1u};
    for (int f = 0; f < m; ++f)
      for (int p = 0; p < pts; ++p) {
        fv_raw[f*pts+p] = float(f*pts+p+1)*0.1f;
        gv_raw[f*pts+p] = F4(float(f+1)*0.2f, float(p+1)*0.1f, 0.f, 0.f);
      }
    auto fv = make_fv(m, pts, fv_raw);
    auto gv = make_gv(m, pts, gv_raw);
    auto da = ref_cpu_density_derivs<F4>(rmm, m, fv, gv, nuc, nuc_count, pts);
    auto db = ref_cpu_density_derivs<F4>(rmm, m, fv, gv, nuc, nuc_count, pts);
    bool ok = true;
    int cdim = COALESCED_DIMENSION(pts);
    for (int na = 0; na < nuc_count; ++na)
      for (int p = 0; p < pts; ++p)
        ok &= near4(da[cdim*na+p], db[cdim*na+p], tol);
    runner.check(ok, "Ra==Rb → deriv_a == deriv_b");
  }

  return runner.summary();
}
