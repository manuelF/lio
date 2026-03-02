// Unit tests for g2g/cuda/kernels/force.h
//
// gpu_compute_forces (closed-shell):
//   forces[atom] = sum_p density_deriv[COALESCED_DIM(pts)*atom + p] * force_factors[p]
//
// gpu_compute_forces_open (open-shell):
//   forces_a[atom] = sum_p density_deriv_a[COALESCED_DIM(pts)*atom + p] * force_factors_a[p]
//   forces_b[atom] = sum_p density_deriv_b[COALESCED_DIM(pts)*atom + p] * force_factors_b[p]
//
// Both kernels use Block=dim3(FORCE_BLOCK_SIZE=256), Grid=dim3(ceil(n_atoms/256)).
// force_factors are loaded into shared memory in chunks of FORCE_BLOCK_SIZE.
//
// Test coverage
// -------------
//   1. Zero factors → all forces (0,0,0,0)
//   2. Single atom, single point → hand-verifiable
//   3. n_atoms=3, pts=5 → vs CPU reference
//   4. pts=300 > FORCE_BLOCK_SIZE=256 → multi-chunk, vs CPU
//   5. pts=513, partial last chunk → vs CPU
//   6. Open-shell: alpha==beta → forces_a == forces_b
//   7. Open-shell: asymmetric alpha/beta → vs CPU

#define GPU_KERNELS 1
#define FULL_DOUBLE 0
#define CPU_KERNELS 0
#define USE_LIBXC   0

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "test_utils.h"
#include "../../../g2g/common.h"               // FORCE_BLOCK_SIZE
#include "../../../g2g/matrix.h"               // COALESCED_DIMENSION
#include "../../../g2g/scalar_vector_types.h"  // vec_type<T,N>
#include "../../../g2g/cuda/cuda_extra.h"      // operator+=, index_x

namespace G2G {
#include "../../../g2g/cuda/kernels/force.h"
}

using F4 = G2G::vec_type<float, 4>;

// ---------------------------------------------------------------------------
// CPU reference: forces[atom] = sum_p derivs[COALESCED_DIM(pts)*atom+p]*f[p]
// derivs must be padded to COALESCED_DIMENSION(pts)*n_atoms elements.
// ---------------------------------------------------------------------------
static std::vector<F4> cpu_forces(int n_atoms, int pts,
                                  const std::vector<float>& factors,
                                  const std::vector<F4>& derivs) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4> out(n_atoms, F4(0.f, 0.f, 0.f, 0.f));
  for (int a = 0; a < n_atoms; ++a) {
    float ax = 0.f, ay = 0.f, az = 0.f, aw = 0.f;
    for (int p = 0; p < pts; ++p) {
      float f   = factors[p];
      F4    d   = derivs[cdim * a + p];
      ax += d.x * f;
      ay += d.y * f;
      az += d.z * f;
      aw += d.w * f;
    }
    out[a] = F4(ax, ay, az, aw);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Run closed-shell force kernel; returns host forces.
// h_derivs must have COALESCED_DIMENSION(pts)*n_atoms elements.
// ---------------------------------------------------------------------------
static std::vector<F4> run_forces(int n_atoms, int pts,
                                  const std::vector<float>& h_factors,
                                  const std::vector<F4>&    h_derivs) {
  int cdim = COALESCED_DIMENSION(pts);
  float* d_factors;
  F4 *d_derivs, *d_forces;
  CUDA_CHECK(cudaMalloc(&d_factors, pts * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_derivs,  (size_t)cdim * n_atoms * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_forces,  n_atoms * sizeof(F4)));
  CUDA_CHECK(cudaMemcpy(d_factors, h_factors.data(), pts * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_derivs,  h_derivs.data(),
                        (size_t)cdim * n_atoms * sizeof(F4),
                        cudaMemcpyHostToDevice));

  dim3 block(FORCE_BLOCK_SIZE);
  dim3 grid((n_atoms + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE);
  G2G::gpu_compute_forces<float><<<grid, block>>>(
      pts, d_factors, d_derivs, d_forces, n_atoms);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<F4> h(n_atoms);
  CUDA_CHECK(cudaMemcpy(h.data(), d_forces, n_atoms * sizeof(F4),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_factors);
  cudaFree(d_derivs);
  cudaFree(d_forces);
  return h;
}

// ---------------------------------------------------------------------------
// Run open-shell force kernel; returns {forces_a, forces_b}.
// ---------------------------------------------------------------------------
static std::pair<std::vector<F4>, std::vector<F4>>
run_forces_open(int n_atoms, int pts,
                const std::vector<float>& h_fa, const std::vector<float>& h_fb,
                const std::vector<F4>&    h_da, const std::vector<F4>&    h_db) {
  int cdim = COALESCED_DIMENSION(pts);
  float *d_fa, *d_fb;
  F4    *d_da, *d_db, *d_forces_a, *d_forces_b;
  CUDA_CHECK(cudaMalloc(&d_fa,      pts * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fb,      pts * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_da,      (size_t)cdim * n_atoms * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_db,      (size_t)cdim * n_atoms * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_forces_a, n_atoms * sizeof(F4)));
  CUDA_CHECK(cudaMalloc(&d_forces_b, n_atoms * sizeof(F4)));
  CUDA_CHECK(cudaMemcpy(d_fa, h_fa.data(), pts * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fb, h_fb.data(), pts * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_da, h_da.data(), (size_t)cdim * n_atoms * sizeof(F4), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_db, h_db.data(), (size_t)cdim * n_atoms * sizeof(F4), cudaMemcpyHostToDevice));

  dim3 block(FORCE_BLOCK_SIZE);
  dim3 grid((n_atoms + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE);
  G2G::gpu_compute_forces_open<float><<<grid, block>>>(
      pts, d_fa, d_fb, d_da, d_db, d_forces_a, d_forces_b, n_atoms);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<F4> ha(n_atoms), hb(n_atoms);
  CUDA_CHECK(cudaMemcpy(ha.data(), d_forces_a, n_atoms * sizeof(F4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(), d_forces_b, n_atoms * sizeof(F4), cudaMemcpyDeviceToHost));
  cudaFree(d_fa); cudaFree(d_fb);
  cudaFree(d_da); cudaFree(d_db);
  cudaFree(d_forces_a); cudaFree(d_forces_b);
  return {ha, hb};
}

static bool near4(F4 a, F4 b, float tol = 2e-4f) {
  return fabsf(a.x-b.x) <= tol && fabsf(a.y-b.y) <= tol &&
         fabsf(a.z-b.z) <= tol && fabsf(a.w-b.w) <= tol;
}
static bool all_near(const std::vector<F4>& g, const std::vector<F4>& r, float tol = 2e-4f) {
  for (size_t i = 0; i < r.size(); ++i) if (!near4(g[i], r[i], tol)) return false;
  return true;
}

// Build a COALESCED_DIMENSION(pts)*n_atoms padded derivs array.
// vals[atom][p] is given as a flat n_atoms*pts vector in row-major.
static std::vector<F4> make_derivs(int n_atoms, int pts, const std::vector<F4>& vals) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4> out(cdim * n_atoms, F4(0.f, 0.f, 0.f, 0.f));
  for (int a = 0; a < n_atoms; ++a)
    for (int p = 0; p < pts; ++p)
      out[cdim * a + p] = vals[a * pts + p];
  return out;
}

// ============================================================================
int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  test_utils::TestRunner runner("gpu_compute_forces kernel");

  printf("[ closed-shell ]\n");

  // --- 1. Zero factors → zero forces ---
  {
    int na = 4, pts = 10;
    int cdim = COALESCED_DIMENSION(pts);
    std::vector<float> factors(pts, 0.f);
    std::vector<F4>    derivs(cdim * na, F4(1.f, 2.f, 3.f, 4.f));
    auto got = run_forces(na, pts, factors, derivs);
    bool ok = true;
    for (auto& f : got) ok &= near4(f, F4(0.f, 0.f, 0.f, 0.f));
    runner.check(ok, "zero factors → zero forces");
  }

  // --- 2. Single atom, single point: force = deriv * factor ---
  // Expected: F4(1,2,3,0) * 2.0 = F4(2,4,6,0)
  {
    int na = 1, pts = 1;
    int cdim = COALESCED_DIMENSION(pts);
    std::vector<float> factors = {2.f};
    std::vector<F4>    derivs(cdim, F4(0.f, 0.f, 0.f, 0.f));
    derivs[0] = F4(1.f, 2.f, 3.f, 0.f);
    auto got = run_forces(na, pts, factors, derivs);
    runner.check(near4(got[0], F4(2.f, 4.f, 6.f, 0.f)),
                 "single atom, single point");
  }

  // --- 3. n_atoms=3, pts=5 → vs CPU ---
  {
    int na = 3, pts = 5;
    std::vector<float> factors(pts);
    std::vector<F4>    vals(na * pts);
    for (int p = 0; p < pts; ++p) factors[p] = float(p + 1) * 0.5f;
    for (int a = 0; a < na; ++a)
      for (int p = 0; p < pts; ++p)
        vals[a * pts + p] = F4(float(a * pts + p + 1), float(-(a * pts + p)),
                               float(a + p) * 0.1f, 0.f);
    auto derivs = make_derivs(na, pts, vals);
    auto got = run_forces(na, pts, factors, derivs);
    auto ref = cpu_forces(na, pts, factors, derivs);
    runner.check(all_near(got, ref), "n_atoms=3 pts=5 vs CPU");
  }

  // --- 4. pts=300 > FORCE_BLOCK_SIZE=256: multi-chunk ---
  {
    int na = 2, pts = 300;
    std::vector<float> factors(pts);
    std::vector<F4>    vals(na * pts);
    for (int p = 0; p < pts; ++p) factors[p] = sinf(float(p) * 0.05f);
    for (int a = 0; a < na; ++a)
      for (int p = 0; p < pts; ++p)
        vals[a * pts + p] = F4(cosf(float(p + a)), sinf(float(p + a)),
                               float(p) * 0.01f, 0.f);
    auto derivs = make_derivs(na, pts, vals);
    auto got = run_forces(na, pts, factors, derivs);
    auto ref = cpu_forces(na, pts, factors, derivs);
    runner.check(all_near(got, ref, 3e-4f), "pts=300 multi-chunk vs CPU");
  }

  // --- 5. pts=513, partial last chunk ---
  {
    int na = 3, pts = 513;
    std::vector<float> factors(pts);
    std::vector<F4>    vals(na * pts);
    for (int p = 0; p < pts; ++p) factors[p] = float(p % 7) * 0.1f;
    for (int a = 0; a < na; ++a)
      for (int p = 0; p < pts; ++p)
        vals[a * pts + p] = F4(float(p % 11) + a, float(a + 1), 0.f, 0.f);
    auto derivs = make_derivs(na, pts, vals);
    auto got = run_forces(na, pts, factors, derivs);
    auto ref = cpu_forces(na, pts, factors, derivs);
    runner.check(all_near(got, ref, 3e-4f), "pts=513 partial last chunk vs CPU");
  }

  printf("\n[ open-shell ]\n");

  // --- 6. Open: alpha == beta → forces_a == forces_b ---
  {
    int na = 2, pts = 50;
    std::vector<float> fa(pts), fb(pts);
    std::vector<F4>    vals(na * pts);
    for (int p = 0; p < pts; ++p) fa[p] = fb[p] = float(p + 1) * 0.01f;
    for (int a = 0; a < na; ++a)
      for (int p = 0; p < pts; ++p)
        vals[a * pts + p] = F4(float(a + 1), float(p + 1) * 0.2f, 0.f, 0.f);
    auto da = make_derivs(na, pts, vals);
    auto db = make_derivs(na, pts, vals);   // same as alpha
    auto pf6 = run_forces_open(na, pts, fa, fb, da, db);
    runner.check(all_near(pf6.first, pf6.second), "open-shell alpha==beta → forces equal");
  }

  // --- 7. Open: asymmetric alpha/beta → vs CPU ---
  {
    int na = 2, pts = 80;
    std::vector<float> fa(pts), fb(pts);
    std::vector<F4>    vals_a(na * pts), vals_b(na * pts);
    for (int p = 0; p < pts; ++p) {
      fa[p] = float(p + 1) * 0.02f;
      fb[p] = float(pts - p) * 0.03f;
    }
    for (int a = 0; a < na; ++a)
      for (int p = 0; p < pts; ++p) {
        vals_a[a * pts + p] = F4(float(a * 10 + p) * 0.1f,  1.f, 0.f, 0.f);
        vals_b[a * pts + p] = F4(float(a *  5 + p) * 0.1f, -1.f, 0.f, 0.f);
      }
    auto da = make_derivs(na, pts, vals_a);
    auto db = make_derivs(na, pts, vals_b);
    auto pf7 = run_forces_open(na, pts, fa, fb, da, db);
    auto ref_a = cpu_forces(na, pts, fa, da);
    auto ref_b = cpu_forces(na, pts, fb, db);
    runner.check(all_near(pf7.first, ref_a, 3e-4f) && all_near(pf7.second, ref_b, 3e-4f),
                 "open-shell asymmetric → vs CPU");
  }

  return runner.summary();
}
