// Unit tests for g2g/cuda/kernels/weight.h
//
// gpu_compute_weights implements Becke fuzzy-cell partitioning:
//
//   For each integration point P assigned to atom A:
//     P_i = prod_{j≠i} s(mu_ij)        where mu_ij = (|PA_i|-|PA_j|)/|A_iA_j|
//     s(u) = 0.5*(1 - f(u)),  f is a 3-step polynomial smoother
//     weight = P_A / sum_i P_i
//
// Tests
// -----
//   1. Single atom, single point       → weight = 1.0
//   2. Single atom, many points        → all weights = 1.0
//   3. Two equal-rm atoms, midpoint    → weight = 0.5
//   4. Two atoms, point at atom 0      → weight = 1.0
//   5. GPU vs CPU reference: two atoms, multiple points
//   6. GPU vs CPU reference: three atoms, multiple points
//   7. Points = 0 (empty)              → no crash

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC 0

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../../g2g/common.h"               // WEIGHT_BLOCK_SIZE, MAX_ATOMS
#include "../../../g2g/cuda/cuda_extra.h"      // ::distance(), index_x()
#include "../../../g2g/scalar_vector_types.h"  // vec_type<T,N>
#include "test_utils.h"

namespace G2G {
#include "../../../g2g/cuda/gpu_variables.h"  // __constant__ gpu_atoms
#include "../../../g2g/cuda/kernels/weight.h"
}  // namespace G2G

// ---------------------------------------------------------------------------
// CPU reference: Becke weight for a single point.
//   atoms[i] = {x, y, z, rm}   — indexed by global atom index
//   nucleii[n] (size nucleii_count) — global atom indices in this partition
//   atom_of_point — which atom the grid point is assigned to
// ---------------------------------------------------------------------------
struct Atom {
  float x, y, z, rm;
};

static float becke_smooth(float u) {
  for (int k = 0; k < 3; k++) u = 1.5f * u - 0.5f * (u * u * u);
  return 0.5f * (1.0f - u);
}

static float cpu_becke_weight(float px, float py, float pz, uint atom_of_point,
                              const std::vector<Atom>& atoms,
                              const std::vector<uint>& nucleii) {
  uint nucleii_count = (uint)nucleii.size();
  float P_total = 0.0f, P_atom = 0.0f;
  bool found = false;

  for (uint ni = 0; ni < nucleii_count; ni++) {
    uint ai = nucleii[ni];
    float dxi = px - atoms[ai].x;
    float dyi = py - atoms[ai].y;
    float dzi = pz - atoms[ai].z;
    float di = sqrtf(dxi * dxi + dyi * dyi + dzi * dzi);
    float P_curr = 1.0f;

    for (uint nj = 0; nj < nucleii_count; nj++) {
      uint aj = nucleii[nj];
      if (ai == aj) continue;

      float dxj = px - atoms[aj].x;
      float dyj = py - atoms[aj].y;
      float dzj = pz - atoms[aj].z;
      float dj = sqrtf(dxj * dxj + dyj * dyj + dzj * dzj);

      float dxij = atoms[ai].x - atoms[aj].x;
      float dyij = atoms[ai].y - atoms[aj].y;
      float dzij = atoms[ai].z - atoms[aj].z;
      float dij = sqrtf(dxij * dxij + dyij * dyij + dzij * dzij);

      float u = (di - dj) / dij;
      float x = atoms[ai].rm / atoms[aj].rm;
      x = (x - 1.0f) / (x + 1.0f);
      u += (x / (x * x - 1.0f)) * (1.0f - u * u);  // x*x < 1 always → safe

      u = becke_smooth(u);
      P_curr *= u;
    }
    P_total += P_curr;
    if (ai == atom_of_point) {
      P_atom = P_curr;
      found = true;
    }
  }
  if (!found) return 0.0f;  // atom_of_point not in partition (skipped in tests)
  return (P_total == 0.0f) ? 0.0f : P_atom / P_total;
}

// ---------------------------------------------------------------------------
// Launch gpu_compute_weights and compare each weight against CPU reference.
// ---------------------------------------------------------------------------
static bool run_weight_test(const std::vector<Atom>& atoms,
                            const std::vector<uint>& nucleii,
                            const std::vector<float>& px,
                            const std::vector<float>& py,
                            const std::vector<float>& pz,
                            const std::vector<uint>& atom_of_point,
                            float tol = 1e-5f) {
  uint n_atoms = (uint)atoms.size();
  uint nucleii_count = (uint)nucleii.size();
  uint points = (uint)px.size();

  // Set gpu_atoms constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(G2G::gpu_atoms, &n_atoms, sizeof(uint)));

  // Build host-side device buffers
  std::vector<G2G::vec_type<float, 4>> h_atom_rm(n_atoms);
  for (uint i = 0; i < n_atoms; i++)
    h_atom_rm[i] = G2G::vec_type<float, 4>(atoms[i].x, atoms[i].y, atoms[i].z,
                                           atoms[i].rm);

  std::vector<G2G::vec_type<float, 4>> h_pos(points);
  for (uint p = 0; p < points; p++)
    h_pos[p] =
        G2G::vec_type<float, 4>(px[p], py[p], pz[p], float(atom_of_point[p]));

  // Device allocations
  G2G::vec_type<float, 4>* d_atom_rm = nullptr;
  G2G::vec_type<float, 4>* d_pos = nullptr;
  float* d_weights = nullptr;
  uint* d_nucleii = nullptr;

  CUDA_CHECK(cudaMalloc(&d_atom_rm, n_atoms * sizeof(G2G::vec_type<float, 4>)));
  CUDA_CHECK(cudaMalloc(&d_pos, points * sizeof(G2G::vec_type<float, 4>)));
  CUDA_CHECK(cudaMalloc(&d_weights, points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_nucleii, nucleii_count * sizeof(uint)));

  CUDA_CHECK(cudaMemcpy(d_atom_rm, h_atom_rm.data(),
                        n_atoms * sizeof(G2G::vec_type<float, 4>),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(),
                        points * sizeof(G2G::vec_type<float, 4>),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nucleii, nucleii.data(), nucleii_count * sizeof(uint),
                        cudaMemcpyHostToDevice));

  // Launch
  dim3 block(WEIGHT_BLOCK_SIZE);
  dim3 grid((points + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE);
  G2G::gpu_compute_weights<float><<<grid, block>>>(
      points, d_pos, d_atom_rm, d_weights, d_nucleii, nucleii_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_weights(points);
  CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights, points * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_atom_rm);
  cudaFree(d_pos);
  cudaFree(d_weights);
  cudaFree(d_nucleii);

  // Verify against CPU reference
  for (uint p = 0; p < points; p++) {
    float ref =
        cpu_becke_weight(px[p], py[p], pz[p], atom_of_point[p], atoms, nucleii);
    float got = h_weights[p];
    if (std::abs(got - ref) > tol * (1.0f + std::abs(ref))) {
      printf("    MISMATCH point %u: expected %.6g  got %.6g\n", p, (double)ref,
             (double)got);
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Convenience builder: check that all GPU weights equal a fixed expected value.
// ---------------------------------------------------------------------------
static bool check_all_equal(const std::vector<float>& h_weights, float expected,
                            float tol, const char* label) {
  for (uint p = 0; p < (uint)h_weights.size(); p++) {
    if (std::abs(h_weights[p] - expected) > tol) {
      printf("    %s: point %u expected %.4f  got %.6g\n", label, p,
             (double)expected, (double)h_weights[p]);
      return false;
    }
  }
  return true;
}

// Launch kernel and return weights (no CPU comparison).
static std::vector<float> gpu_weights(const std::vector<Atom>& atoms,
                                      const std::vector<uint>& nucleii,
                                      const std::vector<float>& px,
                                      const std::vector<float>& py,
                                      const std::vector<float>& pz,
                                      const std::vector<uint>& atom_of_point) {
  uint n_atoms = (uint)atoms.size();
  uint nucleii_count = (uint)nucleii.size();
  uint points = (uint)px.size();

  CUDA_CHECK(cudaMemcpyToSymbol(G2G::gpu_atoms, &n_atoms, sizeof(uint)));

  std::vector<G2G::vec_type<float, 4>> h_atom_rm(n_atoms);
  for (uint i = 0; i < n_atoms; i++)
    h_atom_rm[i] = G2G::vec_type<float, 4>(atoms[i].x, atoms[i].y, atoms[i].z,
                                           atoms[i].rm);

  std::vector<G2G::vec_type<float, 4>> h_pos(points);
  for (uint p = 0; p < points; p++)
    h_pos[p] =
        G2G::vec_type<float, 4>(px[p], py[p], pz[p], float(atom_of_point[p]));

  G2G::vec_type<float, 4>* d_atom_rm = nullptr;
  G2G::vec_type<float, 4>* d_pos = nullptr;
  float* d_weights = nullptr;
  uint* d_nucleii = nullptr;

  CUDA_CHECK(cudaMalloc(&d_atom_rm, n_atoms * sizeof(G2G::vec_type<float, 4>)));
  CUDA_CHECK(cudaMalloc(&d_pos, points * sizeof(G2G::vec_type<float, 4>)));
  CUDA_CHECK(cudaMalloc(&d_weights, points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_nucleii, nucleii_count * sizeof(uint)));

  CUDA_CHECK(cudaMemcpy(d_atom_rm, h_atom_rm.data(),
                        n_atoms * sizeof(G2G::vec_type<float, 4>),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(),
                        points * sizeof(G2G::vec_type<float, 4>),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nucleii, nucleii.data(), nucleii_count * sizeof(uint),
                        cudaMemcpyHostToDevice));

  dim3 block(WEIGHT_BLOCK_SIZE);
  dim3 grid(std::max(1u, (points + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE));
  G2G::gpu_compute_weights<float><<<grid, block>>>(
      points, d_pos, d_atom_rm, d_weights, d_nucleii, nucleii_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_weights(points);
  if (points > 0)
    CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights, points * sizeof(float),
                          cudaMemcpyDeviceToHost));

  cudaFree(d_atom_rm);
  cudaFree(d_pos);
  cudaFree(d_weights);
  cudaFree(d_nucleii);
  return h_weights;
}

int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_compute_weights kernel");

  // --- Test 1: single atom, single point → weight = 1 ---
  {
    std::vector<Atom> atoms = {{0.f, 0.f, 0.f, 1.f}};
    std::vector<uint> nucleii = {0};
    auto w = gpu_weights(atoms, nucleii, {1.f}, {0.f}, {0.f}, {0});
    runner.check(check_all_equal(w, 1.0f, 1e-6f, "single-atom"),
                 "single atom, single point → weight=1");
  }

  // --- Test 2: single atom, many points → all weights = 1 ---
  {
    uint N = 300;
    std::vector<Atom> atoms = {{0.f, 0.f, 0.f, 1.f}};
    std::vector<uint> nucleii = {0};
    std::vector<float> px(N), py(N), pz(N);
    std::vector<uint> aop(N, 0u);
    for (uint p = 0; p < N; p++) {
      px[p] = float(p) * 0.1f;
      py[p] = float(p) * 0.05f;
      pz[p] = 0.f;
    }
    auto w = gpu_weights(atoms, nucleii, px, py, pz, aop);
    runner.check(check_all_equal(w, 1.0f, 1e-6f, "single-atom-many"),
                 "single atom, 300 points → all weights=1");
  }

  // --- Test 3: two equal-rm atoms, point at midpoint → weight = 0.5 ---
  // Atoms at (0,0,0) and (2,0,0) with rm=1; point at (1,0,0).
  // By symmetry, Becke weight = 0.5 exactly.
  {
    std::vector<Atom> atoms = {{0.f, 0.f, 0.f, 1.f}, {2.f, 0.f, 0.f, 1.f}};
    std::vector<uint> nucleii = {0, 1};
    auto w = gpu_weights(atoms, nucleii, {1.f}, {0.f}, {0.f}, {0u});
    runner.check(check_all_equal(w, 0.5f, 1e-5f, "midpoint"),
                 "two atoms, midpoint → weight=0.5");
  }

  // --- Test 4: two atoms, point at atom 0 → weight = 1 ---
  // Atoms at (0,0,0) and (2,0,0); point at (0,0,0).
  // s(-1) = 0.5*(1-(-1)) = 1 → P_atom=1; s(+1)=0 → P_other=0 → weight=1.
  {
    std::vector<Atom> atoms = {{0.f, 0.f, 0.f, 1.f}, {2.f, 0.f, 0.f, 1.f}};
    std::vector<uint> nucleii = {0, 1};
    auto w = gpu_weights(atoms, nucleii, {0.f}, {0.f}, {0.f}, {0u});
    runner.check(check_all_equal(w, 1.0f, 1e-5f, "at-atom"),
                 "two atoms, point at atom 0 → weight=1");
  }

  // --- Tests 5,6: GPU vs CPU reference ---
  printf("\n[ GPU vs CPU reference ]\n");

  // Test 5: two atoms, multiple points along the axis
  {
    std::vector<Atom> atoms = {{0.f, 0.f, 0.f, 1.f}, {4.f, 0.f, 0.f, 1.5f}};
    std::vector<uint> nucleii = {0, 1};
    uint N = 200;
    std::vector<float> px(N), py(N), pz(N);
    std::vector<uint> aop(N);
    for (uint p = 0; p < N; p++) {
      px[p] = float(p) / float(N) * 4.0f + 0.2f;  // avoid exact atom positions
      py[p] = 0.1f;
      pz[p] = 0.05f;
      aop[p] = (px[p] < 2.0f) ? 0u : 1u;  // assign to nearest atom
    }
    runner.check(run_weight_test(atoms, nucleii, px, py, pz, aop),
                 "two atoms, 200 points, varied rm");
  }

  // Test 6: three atoms, points near each
  {
    std::vector<Atom> atoms = {
        {0.f, 0.f, 0.f, 1.0f}, {3.f, 0.f, 0.f, 1.2f}, {1.5f, 2.598f, 0.f, 0.8f}
        // equilateral triangle, side≈3
    };
    std::vector<uint> nucleii = {0, 1, 2};
    // Spread some points near each atom
    std::vector<float> px = {0.3f, 0.2f, 2.8f, 3.1f, 1.5f,
                             1.6f, 1.0f, 2.0f, 1.5f, 0.5f};
    std::vector<float> py = {0.1f, 0.2f, 0.1f, 0.2f, 2.2f,
                             2.4f, 0.5f, 0.5f, 1.0f, 1.0f};
    std::vector<float> pz(10, 0.05f);
    std::vector<uint> aop = {0, 0, 1, 1, 2, 2, 0, 1, 2, 0};
    runner.check(run_weight_test(atoms, nucleii, px, py, pz, aop),
                 "three atoms, 10 points");
  }

  // --- Test 7: empty (points=0) — no crash ---
  {
    std::vector<Atom> atoms = {{0.f, 0.f, 0.f, 1.f}};
    std::vector<uint> nucleii = {0};
    auto w = gpu_weights(atoms, nucleii, {}, {}, {}, {});
    runner.check(w.empty(), "points=0 → no crash, empty result");
  }

  return runner.summary();
}
