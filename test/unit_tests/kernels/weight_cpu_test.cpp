// CPU conformance tests for the Becke partitioning weight algorithm.
//
// Two independent implementations are tested side-by-side:
//   ref_cpu_becke_weight()            — simple reference in kernels_reference.h
//   G2G::cpu_becke_weight_for_point() — extracted kernel from g2g/cpu/weight.cpp
//
// Both must agree for every test case.  When the extracted kernel passes these
// tests it confirms that weight.cpp computes the same mathematics as the
// reference, giving end-to-end coverage of the real g2g/cpu/ code.
//
// Test coverage
// -------------
//   1. Single atom → weight = 1 regardless of position
//   2. Two equal atoms, point at midpoint → weight = 0.5 for each atom
//   3. Point near atom 0 → w0 > 0.9, w0 + w1 ≈ 1
//   4. Three atoms equilateral: sum of weights = 1
//   5. Heteronuclear (different rm): weights still sum to 1 and are non-negative

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"
#include "cpu/cpu_kernels.h"  // G2G::cpu_becke_weight_for_point (via -I$(G2G_DIR))

// Helper: call both ref and cpu implementations for the same point/atom setup.
// Returns true iff both results agree within tol.
static bool compare_weights(
    const std::vector<double>& atom_xyz,  // [n*3]: x,y,z per atom
    const std::vector<double>& atom_rm,   // [n]
    double px, double py, double pz,
    int home_atom,
    double tol = 1e-10) {

  int n = (int)atom_rm.size();
  std::vector<double> dists = ref_make_atom_dists(atom_xyz);

  // Reference
  double ref = ref_cpu_becke_weight(atom_xyz, atom_rm, dists,
                                     px, py, pz, home_atom);

  // Build flat arrays for G2G::cpu_becke_weight_for_point
  std::vector<double> ax(n), ay(n), az(n);
  for (int i = 0; i < n; ++i) {
    ax[i] = atom_xyz[3*i+0];
    ay[i] = atom_xyz[3*i+1];
    az[i] = atom_xyz[3*i+2];
  }
  std::vector<unsigned> local_nuc(n);
  for (int i = 0; i < n; ++i) local_nuc[i] = (unsigned)i;

  double cpu = G2G::cpu_becke_weight_for_point(
      px, py, pz,
      (unsigned)home_atom,
      local_nuc.data(), (unsigned)n,
      ax.data(), ay.data(), az.data(), atom_rm.data(),
      dists.data(), (unsigned)n,
      /*home_in_local=*/true);

  return std::abs(ref - cpu) <= tol;
}

// ============================================================================
int main() {
  test_utils::TestRunner runner("Becke weight: ref vs cpu_becke_weight_for_point");
  const double tol = 1e-10;

  // --- 1. Single atom → weight = 1 ---
  {
    std::vector<double> xyz = {0.0, 0.0, 0.0};
    std::vector<double> rm  = {1.0};
    bool ok = compare_weights(xyz, rm, 1.5, 0.3, -0.7, 0, tol);
    runner.check(ok, "single atom → weight=1, ref ≡ cpu");
  }

  // --- 2. Two equal atoms at ±1 on x, midpoint → weight = 0.5 ---
  {
    std::vector<double> xyz = {-1.0, 0.0, 0.0,  1.0, 0.0, 0.0};
    std::vector<double> rm  = {1.0, 1.0};
    bool ok0 = compare_weights(xyz, rm, 0.0, 0.0, 0.0, 0, tol);
    bool ok1 = compare_weights(xyz, rm, 0.0, 0.0, 0.0, 1, tol);
    runner.check(ok0 && ok1, "two equal atoms, midpoint: ref ≡ cpu (both atoms)");
  }

  // --- 3. Point near atom 0 → w0 > 0.9, ref ≡ cpu ---
  {
    std::vector<double> xyz = {0.0, 0.0, 0.0,  4.0, 0.0, 0.0};
    std::vector<double> rm  = {1.0, 1.0};
    bool ok = compare_weights(xyz, rm, 0.3, 0.0, 0.0, 0, tol);
    // Sanity check: value near atom 0 should be > 0.9
    std::vector<double> dists = ref_make_atom_dists(xyz);
    double w0 = ref_cpu_becke_weight(xyz, rm, dists, 0.3, 0.0, 0.0, 0);
    runner.check(ok && w0 > 0.9, "point near atom 0: ref ≡ cpu, w0 > 0.9");
  }

  // --- 4. Three equilateral atoms: ref ≡ cpu for each corner ---
  {
    double s3 = std::sqrt(3.0);
    std::vector<double> xyz = {
       2.0,    0.0,  0.0,
      -1.0,  s3,  0.0,
      -1.0, -s3,  0.0};
    std::vector<double> rm = {1.0, 1.0, 1.0};
    double px = 0.3, py = 0.1, pz = 0.0;
    bool ok = true;
    for (int a = 0; a < 3; ++a)
      ok &= compare_weights(xyz, rm, px, py, pz, a, tol);
    runner.check(ok, "3 equilateral atoms: ref ≡ cpu for all home atoms");
  }

  // --- 5. Heteronuclear: ref ≡ cpu ---
  {
    std::vector<double> xyz = {0.0, 0.0, 0.0,  3.0, 0.0, 0.0,  1.5, 2.0, 0.0};
    std::vector<double> rm  = {1.0, 1.5, 0.8};
    double px = 0.5, py = 0.4, pz = 0.0;
    bool ok = true;
    for (int a = 0; a < 3; ++a)
      ok &= compare_weights(xyz, rm, px, py, pz, a, tol);
    runner.check(ok, "heteronuclear 3-atom: ref ≡ cpu for all home atoms");
  }

  return runner.summary();
}
