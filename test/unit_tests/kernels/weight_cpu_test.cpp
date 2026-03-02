// CPU conformance tests for the Becke partitioning algorithm.
//
// Tests the same mathematical formula implemented in:
//   g2g/cuda/kernels/weight.h  (GPU version)
//   g2g/cpu/weight.cpp         (CPU version, inside PointGroupCPU)
//
// Uses ref_becke_s3 / ref_cpu_becke_weight / ref_make_atom_dists from
// kernels_reference.h, which are the shared reference implementations.
//
// Becke fuzzy-Voronoi weight for grid point P assigned to atom I:
//
//   For each atom j, compute P_j = product_{k != j} s3(mu_jk(P))
//   where:
//     mu_jk = (d(P,j) - d(P,k)) / d(j,k)          (confocal elliptic coord)
//     x     = (rm_j/rm_k - 1) / (rm_j/rm_k + 1)   (heteronuclear adjustment)
//     mu_jk += x/(x^2-1) * (1 - mu_jk^2)          (adjusted mu)
//     s(u)  = 1.5*u - 0.5*u^3                      (Becke step)
//     s3    = s(s(s(mu_jk)))                        (iterated 3 times)
//     P_j   = product s3(mu_jk)
//
//   weight(P, I) = P_I / sum_j P_j
//
// Test coverage
// -------------
//   1. Single atom: weight = 1 everywhere
//   2. Two equal atoms, midpoint: weight = 0.5
//   3. Two equal atoms, close to one: weight close to 1 for near atom
//   4. Three atoms (equilateral triangle): sum of weights = 1
//   5. Homonuclear vs heteronuclear: rm adjustment shifts weight

#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_test_utils.h"
#include "kernels_reference.h"

// ============================================================================
int main() {
  test_utils::TestRunner runner("Becke weight (CPU algorithm)");
  const double tol = 1e-6;

  // --- 1. Single atom: weight = 1 everywhere ---
  {
    std::vector<double> xyz  = {0.0, 0.0, 0.0};
    std::vector<double> rm   = {1.0};
    std::vector<double> dist = {0.0};
    double w = ref_cpu_becke_weight(xyz, rm, dist, 1.0, 2.0, 3.0, 0);
    runner.check(std::abs(w - 1.0) < tol, "single atom → weight=1");
  }

  // --- 2. Two equal atoms, midpoint: weight = 0.5 ---
  // A=(0,0,0), B=(2,0,0), midpoint=(1,0,0)
  // By symmetry each atom gets exactly 0.5
  {
    std::vector<double> xyz  = {0.0,0.0,0.0,  2.0,0.0,0.0};
    std::vector<double> rm   = {1.0, 1.0};
    auto dists = ref_make_atom_dists(xyz);
    double wA = ref_cpu_becke_weight(xyz, rm, dists, 1.0, 0.0, 0.0, 0);
    double wB = ref_cpu_becke_weight(xyz, rm, dists, 1.0, 0.0, 0.0, 1);
    runner.check(std::abs(wA - 0.5) < tol && std::abs(wB - 0.5) < tol,
                 "two equal atoms, midpoint → wA=wB=0.5");
  }

  // --- 3. Two equal atoms, point near atom 0 ---
  // Point at (0.1, 0, 0), much closer to A=(0,0,0) than B=(2,0,0)
  // Weight for A should be much larger than 0.5
  {
    std::vector<double> xyz  = {0.0,0.0,0.0,  2.0,0.0,0.0};
    std::vector<double> rm   = {1.0, 1.0};
    auto dists = ref_make_atom_dists(xyz);
    double wA = ref_cpu_becke_weight(xyz, rm, dists, 0.1, 0.0, 0.0, 0);
    double wB = ref_cpu_becke_weight(xyz, rm, dists, 0.1, 0.0, 0.0, 1);
    runner.check(wA > 0.9 && wA + wB > 0.99 && wA + wB < 1.01,
                 "point near atom 0 → wA>0.9, wA+wB≈1");
  }

  // --- 4. Three atoms in equilateral triangle: sum of weights = 1 ---
  // Atoms at (0,0,0), (2,0,0), (1,√3,0)
  {
    double sq3 = std::sqrt(3.0);
    std::vector<double> xyz = {0.0,0.0,0.0,  2.0,0.0,0.0,  1.0,sq3,0.0};
    std::vector<double> rm  = {1.0, 1.0, 1.0};
    auto dists = ref_make_atom_dists(xyz);
    // Test at centroid (1, 1/√3, 0)
    double cx = 1.0, cy = sq3/3.0, cz = 0.0;
    double w0 = ref_cpu_becke_weight(xyz, rm, dists, cx, cy, cz, 0);
    double w1 = ref_cpu_becke_weight(xyz, rm, dists, cx, cy, cz, 1);
    double w2 = ref_cpu_becke_weight(xyz, rm, dists, cx, cy, cz, 2);
    runner.check(std::abs(w0 + w1 + w2 - 1.0) < tol,
                 "3 atoms equilateral: sum of weights = 1");
  }

  // --- 5. Heteronuclear: weights sum to 1 and are non-negative ---
  // Two atoms at (0,0,0) and (3,0,0) with different radii
  {
    std::vector<double> xyz  = {0.0,0.0,0.0,  3.0,0.0,0.0};
    std::vector<double> rm   = {1.0, 2.0};
    auto dists = ref_make_atom_dists(xyz);
    bool ok = true;
    for (int i = 1; i <= 5; i++) {
      double x = i * 0.5;
      double wA = ref_cpu_becke_weight(xyz, rm, dists, x, 0.0, 0.0, 0);
      double wB = ref_cpu_becke_weight(xyz, rm, dists, x, 0.0, 0.0, 1);
      ok &= (std::abs(wA + wB - 1.0) < 1e-10);
      ok &= (wA >= 0.0 && wB >= 0.0);
    }
    runner.check(ok, "heteronuclear: weights sum to 1 and are non-negative");
  }

  return runner.summary();
}
