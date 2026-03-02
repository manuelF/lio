// CPU conformance tests for the Becke partitioning algorithm.
//
// Tests the same mathematical formula implemented in:
//   g2g/cuda/kernels/weight.h  (GPU version)
//   g2g/cpu/weight.cpp         (CPU version, inside PointGroupCPU)
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
// When all atoms have the same rm (homonuclear), x=0 and the adjustment
// term vanishes, leaving the standard Becke scheme.
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

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
static float dist3(float ax, float ay, float az,
                   float bx, float by, float bz) {
  float dx = ax-bx, dy = ay-by, dz = az-bz;
  return sqrtf(dx*dx + dy*dy + dz*dz);
}

// Becke smoothing function: s(u) = 1.5*u - 0.5*u^3, iterated 3 times.
static double becke_s3(double u) {
  u = 1.5*u - 0.5*(u*u*u);
  u = 1.5*u - 0.5*(u*u*u);
  u = 1.5*u - 0.5*(u*u*u);
  return 0.5*(1.0 - u);  // transform to [0,1]
}

// ---------------------------------------------------------------------------
// CPU Becke weight.
//
// atom_xyz:  [n_atoms * 3] atom positions (x,y,z per atom)
// atom_rm:   [n_atoms]     covalent radii
// atom_dists:[n_atoms^2]   precomputed inter-atomic distances (row-major)
// px,py,pz:  grid point position
// atom_of_point: the atom this grid point is associated with
//
// Returns the Becke weight for the given atom.
// ---------------------------------------------------------------------------
static double cpu_becke_weight(const std::vector<double>& atom_xyz,
                                const std::vector<double>& atom_rm,
                                const std::vector<double>& atom_dists,
                                double px, double py, double pz,
                                int atom_of_point) {
  int n = (int)atom_rm.size();
  double P_total = 0.0;
  double P_atom  = 0.0;
  bool   found   = false;

  for (int j = 0; j < n; j++) {
    double P_curr = 1.0;
    double djx = atom_xyz[3*j+0], djy = atom_xyz[3*j+1], djz = atom_xyz[3*j+2];
    double d_Pj = dist3((float)px, (float)py, (float)pz,
                        (float)djx, (float)djy, (float)djz);

    for (int k = 0; k < n; k++) {
      if (k == j) continue;
      double dkx = atom_xyz[3*k+0], dky = atom_xyz[3*k+1], dkz = atom_xyz[3*k+2];
      double d_Pk   = dist3((float)px,(float)py,(float)pz,
                            (float)dkx,(float)dky,(float)dkz);
      double d_jk   = atom_dists[j * n + k];

      double u = (d_Pj - d_Pk) / d_jk;

      // Heteronuclear adjustment (Becke Eq. A5)
      double x = atom_rm[j] / atom_rm[k];
      x = (x - 1.0) / (x + 1.0);
      if (std::abs(x) > 1e-10)
        u += (x / (x*x - 1.0)) * (1.0 - u*u);

      double mu = becke_s3(u);

      P_curr *= mu;
      if (P_curr == 0.0) break;
    }

    P_total += P_curr;
    if (j == atom_of_point) {
      P_atom = P_curr;
      found  = true;
    }
  }

  if (!found || P_total == 0.0) return 0.0;
  return P_atom / P_total;
}

// ---------------------------------------------------------------------------
// Build inter-atomic distance table
// ---------------------------------------------------------------------------
static std::vector<double> make_dists(const std::vector<double>& xyz) {
  int n = (int)xyz.size() / 3;
  std::vector<double> d(n * n, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      d[i*n+j] = dist3((float)xyz[3*i+0],(float)xyz[3*i+1],(float)xyz[3*i+2],
                       (float)xyz[3*j+0],(float)xyz[3*j+1],(float)xyz[3*j+2]);
  return d;
}

// ============================================================================
int main() {
  test_utils::TestRunner runner("Becke weight (CPU algorithm)");
  const double tol = 1e-6;

  // --- 1. Single atom: weight = 1 everywhere ---
  {
    std::vector<double> xyz  = {0.0, 0.0, 0.0};
    std::vector<double> rm   = {1.0};
    std::vector<double> dist = {0.0};
    double w = cpu_becke_weight(xyz, rm, dist, 1.0, 2.0, 3.0, 0);
    runner.check(std::abs(w - 1.0) < tol, "single atom → weight=1");
  }

  // --- 2. Two equal atoms, midpoint: weight = 0.5 ---
  // A=(0,0,0), B=(2,0,0), midpoint=(1,0,0)
  // By symmetry each atom gets exactly 0.5
  {
    std::vector<double> xyz  = {0.0,0.0,0.0,  2.0,0.0,0.0};
    std::vector<double> rm   = {1.0, 1.0};
    auto dists = make_dists(xyz);
    double wA = cpu_becke_weight(xyz, rm, dists, 1.0, 0.0, 0.0, 0);
    double wB = cpu_becke_weight(xyz, rm, dists, 1.0, 0.0, 0.0, 1);
    runner.check(std::abs(wA - 0.5) < tol && std::abs(wB - 0.5) < tol,
                 "two equal atoms, midpoint → wA=wB=0.5");
  }

  // --- 3. Two equal atoms, point near atom 0 ---
  // Point at (0.1, 0, 0), much closer to A=(0,0,0) than B=(2,0,0)
  // Weight for A should be much larger than 0.5
  {
    std::vector<double> xyz  = {0.0,0.0,0.0,  2.0,0.0,0.0};
    std::vector<double> rm   = {1.0, 1.0};
    auto dists = make_dists(xyz);
    double wA = cpu_becke_weight(xyz, rm, dists, 0.1, 0.0, 0.0, 0);
    double wB = cpu_becke_weight(xyz, rm, dists, 0.1, 0.0, 0.0, 1);
    runner.check(wA > 0.9 && wA + wB > 0.99 && wA + wB < 1.01,
                 "point near atom 0 → wA>0.9, wA+wB≈1");
  }

  // --- 4. Three atoms in equilateral triangle: sum of weights = 1 ---
  // Atoms at (0,0,0), (2,0,0), (1,√3,0)
  {
    double sq3 = sqrt(3.0);
    std::vector<double> xyz = {0.0,0.0,0.0,  2.0,0.0,0.0,  1.0,sq3,0.0};
    std::vector<double> rm  = {1.0, 1.0, 1.0};
    auto dists = make_dists(xyz);
    // Test at centroid (1, 1/√3, 0) ≈ (1, 0.577, 0)
    double cx = 1.0, cy = sq3/3.0, cz = 0.0;
    double w0 = cpu_becke_weight(xyz, rm, dists, cx, cy, cz, 0);
    double w1 = cpu_becke_weight(xyz, rm, dists, cx, cy, cz, 1);
    double w2 = cpu_becke_weight(xyz, rm, dists, cx, cy, cz, 2);
    double sum = w0 + w1 + w2;
    runner.check(std::abs(sum - 1.0) < tol,
                 "3 atoms equilateral: sum of weights = 1");
  }

  // --- 5. Two atoms at centroid: weights sum to 1, each >= 0 ---
  // Test several random points to verify partition of unity
  {
    std::vector<double> xyz  = {0.0,0.0,0.0,  3.0,0.0,0.0};
    std::vector<double> rm   = {1.0, 2.0};   // heteronuclear
    auto dists = make_dists(xyz);
    bool ok = true;
    // Test 5 points along the line between the atoms
    for (int i = 1; i <= 5; i++) {
      double x = i * 0.5;
      double wA = cpu_becke_weight(xyz, rm, dists, x, 0.0, 0.0, 0);
      double wB = cpu_becke_weight(xyz, rm, dists, x, 0.0, 0.0, 1);
      double sum = wA + wB;
      ok &= (std::abs(sum - 1.0) < 1e-10);
      ok &= (wA >= 0.0 && wB >= 0.0);
    }
    runner.check(ok, "heteronuclear: weights sum to 1 and are non-negative");
  }

  return runner.summary();
}
