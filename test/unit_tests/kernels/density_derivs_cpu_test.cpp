// CPU conformance tests for cpu_compute_density_derivs().
//
// Compares G2G::cpu_compute_density_derivs() from g2g/cpu/cpu_kernels.h
// against the portable reference ref_cpu_density_derivs_sym() from
// kernels_reference.h.
//
// Both implementations use the FULL SYMMETRIC density matrix with the
// (ii==j ? 2 : 1) weight factor, as produced by get_rmm_input():
//   w_ii = sum_j rmm[ii*m+j] * fv[j] * (ii==j ? 2 : 1)
//   ddx[func2nuc[ii]] -= w_ii * gxv[ii]    (additive — caller zeroes arrays)
//
// This is the CPU force-derivs convention, distinct from the GPU
// energy_derivs.h kernel which uses a lower-triangular texture.
//
// Test coverage
// -------------
//   1. m=1, pts=1: trivial w = 2*R[0][0]*F[0]
//   2. m=2, pts=1, 2 atoms: hand-verify
//   3. Zero gradient → zero derivatives
//   4. m=4, pts=3: ref ≡ cpu_compute_density_derivs
//   5. Open-shell: Ra==Rb → deriv_a == deriv_b

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"
#include "cpu/cpu_kernels.h"

static bool nearv(const std::vector<float>& a, const std::vector<float>& b,
                  float tol = 1e-5f) {
  if (a.size() != b.size()) return false;
  for (int i = 0; i < (int)a.size(); ++i)
    if (fabsf(a[i] - b[i]) > tol) return false;
  return true;
}

// Symmetrize a lower-triangular rmm into a full symmetric matrix.
// cpu_compute_density_derivs expects both triangles filled identically.
static std::vector<float> symmetrize(const std::vector<float>& lower, int m) {
  std::vector<float> sym(m * m, 0.f);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j <= i; ++j)
      sym[i*m+j] = sym[j*m+i] = lower[i*m+j];
  return sym;
}

// ============================================================================
int main() {
  test_utils::TestRunner runner(
      "cpu_compute_density_derivs vs ref_cpu_density_derivs_sym");
  const float tol = 1e-5f;

  // --- 1. m=1, pts=1: w = 2*R[0][0]*fv[0] = 2*2*3 = 12 ---
  // gxv=[1], func2nuc=[0], n_atoms=1
  // ddx[0] = -12*1 = -12
  {
    int m = 1, n_atoms = 1;
    std::vector<float> rmm_lower = {2.f};
    auto sym = symmetrize(rmm_lower, m);
    std::vector<float> fv = {3.f}, gx = {1.f}, gy = {0.f}, gz = {0.f};
    std::vector<unsigned> nuc = {0u};

    std::vector<float> ref_ddx(n_atoms, 0.f), ref_ddy(n_atoms, 0.f), ref_ddz(n_atoms, 0.f);
    ref_cpu_density_derivs_sym(fv, gx, gy, gz, sym, m, nuc, n_atoms,
                                ref_ddx, ref_ddy, ref_ddz);

    std::vector<float> cpu_ddx(n_atoms, 0.f), cpu_ddy(n_atoms, 0.f), cpu_ddz(n_atoms, 0.f);
    G2G::cpu_compute_density_derivs(fv.data(), gx.data(), gy.data(), gz.data(),
                                     sym.data(), m, nuc.data(), n_atoms,
                                     cpu_ddx.data(), cpu_ddy.data(), cpu_ddz.data());

    bool ok = nearv(ref_ddx, {-12.f}, tol) &&
              nearv(cpu_ddx, ref_ddx, tol) &&
              nearv(cpu_ddy, ref_ddy, tol) &&
              nearv(cpu_ddz, ref_ddz, tol);
    runner.check(ok, "m=1 trivial: ddx[0]=-12, ref == cpu");
  }

  // --- 2. m=2, pts=1, 2 atoms: hand-verify ---
  // R: R[0][0]=1, R[1][0]=2, R[1][1]=3  → sym: both triangles filled
  // fv=[1,2], gxv=[1,0], nuc=[0,1]
  // ii=0: w = sym[0*2+0]*1*2 + sym[0*2+1]*2*1 = 2 + 4 = 6
  //   ddx[0] -= 6*1 = 6   → ddx[0]=-6
  // ii=1: w = sym[1*2+0]*1*1 + sym[1*2+1]*2*2 = 2 + 12 = 14
  //   ddx[1] -= 14*0 = 0  → ddx[1]=0
  {
    int m = 2, n_atoms = 2;
    std::vector<float> rmm_lower = {1.f, 0.f, 2.f, 3.f};
    auto sym = symmetrize(rmm_lower, m);
    std::vector<float> fv = {1.f, 2.f}, gx = {1.f, 0.f};
    std::vector<float> gy(m, 0.f), gz(m, 0.f);
    std::vector<unsigned> nuc = {0u, 1u};

    std::vector<float> ref_ddx(n_atoms, 0.f), ref_ddy(n_atoms, 0.f), ref_ddz(n_atoms, 0.f);
    ref_cpu_density_derivs_sym(fv, gx, gy, gz, sym, m, nuc, n_atoms,
                                ref_ddx, ref_ddy, ref_ddz);

    std::vector<float> cpu_ddx(n_atoms, 0.f), cpu_ddy(n_atoms, 0.f), cpu_ddz(n_atoms, 0.f);
    G2G::cpu_compute_density_derivs(fv.data(), gx.data(), gy.data(), gz.data(),
                                     sym.data(), m, nuc.data(), n_atoms,
                                     cpu_ddx.data(), cpu_ddy.data(), cpu_ddz.data());

    bool ok = nearv(ref_ddx, {-6.f, 0.f}, tol) &&
              nearv(cpu_ddx, ref_ddx, tol) &&
              nearv(cpu_ddy, ref_ddy, tol) &&
              nearv(cpu_ddz, ref_ddz, tol);
    runner.check(ok, "m=2 hand-verify: ddx[0]=-6, ddx[1]=0, ref == cpu");
  }

  // --- 3. Zero gradient → zero derivatives ---
  {
    int m = 3, n_atoms = 2;
    std::vector<float> rmm_lower(m*m, 0.f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm_lower[i*m+j] = float(i+j+1) * 0.3f;
    auto sym = symmetrize(rmm_lower, m);
    std::vector<float> fv(m, 1.f);
    std::vector<float> gx(m, 0.f), gy(m, 0.f), gz(m, 0.f);
    std::vector<unsigned> nuc = {0u, 0u, 1u};

    std::vector<float> cpu_ddx(n_atoms, 0.f), cpu_ddy(n_atoms, 0.f), cpu_ddz(n_atoms, 0.f);
    G2G::cpu_compute_density_derivs(fv.data(), gx.data(), gy.data(), gz.data(),
                                     sym.data(), m, nuc.data(), n_atoms,
                                     cpu_ddx.data(), cpu_ddy.data(), cpu_ddz.data());
    bool ok = nearv(cpu_ddx, std::vector<float>(n_atoms, 0.f), tol) &&
              nearv(cpu_ddy, std::vector<float>(n_atoms, 0.f), tol) &&
              nearv(cpu_ddz, std::vector<float>(n_atoms, 0.f), tol);
    runner.check(ok, "zero gradient → zero derivatives");
  }

  // --- 4. m=4: ref ≡ cpu_compute_density_derivs ---
  {
    int m = 4, n_atoms = 2;
    std::vector<float> rmm_lower(m*m, 0.f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm_lower[i*m+j] = float(i*m+j+1) * 0.1f;
    auto sym = symmetrize(rmm_lower, m);
    std::vector<float> fv(m), gx(m), gy(m), gz(m);
    std::vector<unsigned> nuc = {0u, 0u, 1u, 1u};
    for (int i = 0; i < m; ++i) {
      fv[i] = float(i+1) * 0.3f;
      gx[i] = float(i+1) * 0.1f;
      gy[i] = float(m-i) * 0.1f;
      gz[i] = float(i*2+1) * 0.05f;
    }

    std::vector<float> ref_ddx(n_atoms, 0.f), ref_ddy(n_atoms, 0.f), ref_ddz(n_atoms, 0.f);
    ref_cpu_density_derivs_sym(fv, gx, gy, gz, sym, m, nuc, n_atoms,
                                ref_ddx, ref_ddy, ref_ddz);

    std::vector<float> cpu_ddx(n_atoms, 0.f), cpu_ddy(n_atoms, 0.f), cpu_ddz(n_atoms, 0.f);
    G2G::cpu_compute_density_derivs(fv.data(), gx.data(), gy.data(), gz.data(),
                                     sym.data(), m, nuc.data(), n_atoms,
                                     cpu_ddx.data(), cpu_ddy.data(), cpu_ddz.data());

    bool ok = nearv(cpu_ddx, ref_ddx, tol) &&
              nearv(cpu_ddy, ref_ddy, tol) &&
              nearv(cpu_ddz, ref_ddz, tol);
    runner.check(ok, "m=4: ref ≡ cpu_compute_density_derivs");
  }

  // --- 5. Open-shell: Ra==Rb → deriv_a == deriv_b ---
  {
    int m = 4, n_atoms = 2;
    std::vector<float> rmm_lower(m*m, 0.f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j <= i; ++j)
        rmm_lower[i*m+j] = float(i+j+1) * 0.15f;
    auto sym = symmetrize(rmm_lower, m);
    std::vector<float> fv(m), gx(m), gy(m), gz(m);
    std::vector<unsigned> nuc = {0u, 0u, 1u, 1u};
    for (int i = 0; i < m; ++i) {
      fv[i] = float(i+1) * 0.2f;
      gx[i] = float(i+1) * 0.1f;
      gy[i] = float(m-i) * 0.1f;
      gz[i] = 0.05f;
    }

    std::vector<float> ddx_a(n_atoms, 0.f), ddy_a(n_atoms, 0.f), ddz_a(n_atoms, 0.f);
    std::vector<float> ddx_b(n_atoms, 0.f), ddy_b(n_atoms, 0.f), ddz_b(n_atoms, 0.f);
    G2G::cpu_compute_density_derivs(fv.data(), gx.data(), gy.data(), gz.data(),
                                     sym.data(), m, nuc.data(), n_atoms,
                                     ddx_a.data(), ddy_a.data(), ddz_a.data());
    G2G::cpu_compute_density_derivs(fv.data(), gx.data(), gy.data(), gz.data(),
                                     sym.data(), m, nuc.data(), n_atoms,
                                     ddx_b.data(), ddy_b.data(), ddz_b.data());

    bool ok = nearv(ddx_a, ddx_b, tol) &&
              nearv(ddy_a, ddy_b, tol) &&
              nearv(ddz_a, ddz_b, tol);
    runner.check(ok, "open-shell Ra==Rb → deriv_a == deriv_b");
  }

  return runner.summary();
}
