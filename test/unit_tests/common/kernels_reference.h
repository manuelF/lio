#pragma once
// Portable C++ mathematical reference implementations shared by both CUDA
// (*_test.cu) and CPU (*_cpu_test.cpp) unit tests.
//
// CUDA tests compile with nvcc (GPU_KERNELS=1).  CPU tests compile with g++.
// All functions are plain C++14 using only <cmath> and <vector>.
//
// Covered algorithms (matching g2g/cpu/*.cpp and g2g/cuda/kernels/*.h):
//   - GTO basis function values, gradients, and Hessians (functions.cpp)
//   - Becke partition weights                            (weight.cpp)
//   - Electron density from RMM + function values       (energy.h)
//   - Force integration over density derivatives        (force.h)
//   - Density derivatives from RMM + grad functions     (energy_derivs.h)

#include <cmath>
#include <vector>

// COALESCED_DIMENSION(d): smallest multiple of 32 strictly greater than d.
// Matches g2g/matrix.h — defined as a fallback so this header works without it.
#ifndef COALESCED_DIMENSION
#define COALESCED_DIMENSION(d) ((d) + 32 - (d) % 32)
#endif

// ---------------------------------------------------------------------------
// Simple portable vector types
// ---------------------------------------------------------------------------

struct RefF4 {
  float x, y, z, w;
  RefF4() : x(0.f), y(0.f), z(0.f), w(0.f) {}
  RefF4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
};

struct RefVec3 {
  float x, y, z;
  RefVec3(float _x = 0.f, float _y = 0.f, float _z = 0.f) : x(_x), y(_y), z(_z) {}
};

struct RefHess {
  // Diagonal Hessian components: d2phi/dx2, d2phi/dy2, d2phi/dz2
  float px, py, pz;
  // Cross Hessian components: d2phi/dxdy, d2phi/dxdz, d2phi/dydz
  float ix, iy, iz;
  RefHess(float _px=0,float _py=0,float _pz=0,
          float _ix=0,float _iy=0,float _iz=0)
      : px(_px), py(_py), pz(_pz), ix(_ix), iy(_iy), iz(_iz) {}
};

// ============================================================================
// GTO (Gaussian-Type Orbital) basis function evaluation
// Matches g2g/cpu/functions.cpp and g2g/cuda/kernels/functions.h
// ============================================================================

// Radial contraction: t = sum_c coeff_c * exp(-alpha_c * dist2)
// Skips contractions with alpha_c * dist2 > 70 (underflow guard, same as kernel).
inline float ref_gto_radial(float dist2,
                             const std::vector<float>& alphas,
                             const std::vector<float>& coeffs) {
  float t = 0.f;
  for (int c = 0; c < (int)alphas.size(); ++c) {
    float expon = alphas[c] * dist2;
    if (expon > 70.f) continue;
    t += expf(-expon) * coeffs[c];
  }
  return t;
}

// Radial sums needed for gradients and Hessians:
//   t  = sum_c  c * exp(-a*r2)
//   tg = sum_c  c*a * exp(-a*r2)    (first derivative weight)
//   th = sum_c  c*a^2 * exp(-a*r2)  (second derivative weight)
inline void ref_gto_radial_tg_th(float dist2,
                                   const std::vector<float>& alphas,
                                   const std::vector<float>& coeffs,
                                   float& t, float& tg, float& th) {
  t = tg = th = 0.f;
  for (int c = 0; c < (int)alphas.size(); ++c) {
    float a = alphas[c];
    float expon = a * dist2;
    if (expon > 70.f) continue;
    float t0 = expf(-expon) * coeffs[c];
    t  += t0;
    tg += t0 * a;
    th += t0 * (a * a);
  }
}

// GTO value for a single shell component.
// shell_type: 0=S, 1=Px, 2=Py, 3=Pz,
//             4=DXX, 5=DXY(=YX), 6=DYY, 7=DXZ(=ZX), 8=DYZ(=ZY), 9=DZZ
// norm: normalization factor for diagonal D components (XX, YY, ZZ).
// v = point - atom_position, dist2 = |v|^2
inline float ref_eval_gto_value(float vx, float vy, float vz, float dist2,
                                 const std::vector<float>& alphas,
                                 const std::vector<float>& coeffs,
                                 int shell_type, float norm = 1.f) {
  float t = ref_gto_radial(dist2, alphas, coeffs);
  switch (shell_type) {
    case 0: return t;
    case 1: return vx * t;
    case 2: return vy * t;
    case 3: return vz * t;
    case 4: return norm * vx * vx * t;
    case 5: return vy * vx * t;
    case 6: return norm * vy * vy * t;
    case 7: return vz * vx * t;
    case 8: return vz * vy * t;
    case 9: return norm * vz * vz * t;
    default: return 0.f;
  }
}

// GTO gradient: (gx, gy, gz) = grad_r(phi) for the given shell component.
// Matches g2g/cpu/functions.cpp gradient formulas exactly.
inline RefVec3 ref_eval_gto_grad(float vx, float vy, float vz, float dist2,
                                  const std::vector<float>& alphas,
                                  const std::vector<float>& coeffs,
                                  int shell_type, float norm = 1.f) {
  float t, tg, th;
  ref_gto_radial_tg_th(dist2, alphas, coeffs, t, tg, th);
  (void)th;

  // Helper: -2*tg * vi
  auto neg2tg = [&](float vi) { return -2.f * tg * vi; };

  switch (shell_type) {
    // S: phi = t,  grad = -2*tg*(vx, vy, vz)
    case 0:
      return { neg2tg(vx), neg2tg(vy), neg2tg(vz) };

    // Px: phi = vx*t,  d/dk(vx*t) = delta_{kx}*t - 2*tg*vk*vx
    case 1:
      return { t + neg2tg(vx)*vx, neg2tg(vy)*vx, neg2tg(vz)*vx };
    // Py: phi = vy*t
    case 2:
      return { neg2tg(vx)*vy, t + neg2tg(vy)*vy, neg2tg(vz)*vy };
    // Pz: phi = vz*t
    case 3:
      return { neg2tg(vx)*vz, neg2tg(vy)*vz, t + neg2tg(vz)*vz };

    // DXX: phi = norm*vx^2*t
    case 4:
      return { norm*(2.f*vx*t + neg2tg(vx)*vx*vx),
               norm*(            neg2tg(vy)*vx*vx),
               norm*(            neg2tg(vz)*vx*vx) };

    // DXY(=YX): phi = vy*vx*t,  d/dk(vy*vx*t) = delta_{ky}*vx*t + delta_{kx}*vy*t - 2*tg*vk*vy*vx
    case 5:
      return { vy*t + neg2tg(vx)*vy*vx,
               vx*t + neg2tg(vy)*vy*vx,
                      neg2tg(vz)*vy*vx };

    // DYY: phi = norm*vy^2*t
    case 6:
      return { norm*(            neg2tg(vx)*vy*vy),
               norm*(2.f*vy*t + neg2tg(vy)*vy*vy),
               norm*(            neg2tg(vz)*vy*vy) };

    // DXZ(=ZX): phi = vz*vx*t
    case 7:
      return { vz*t + neg2tg(vx)*vz*vx,
                      neg2tg(vy)*vz*vx,
               vx*t + neg2tg(vz)*vz*vx };

    // DYZ(=ZY): phi = vz*vy*t
    case 8:
      return {        neg2tg(vx)*vz*vy,
               vz*t + neg2tg(vy)*vz*vy,
               vy*t + neg2tg(vz)*vz*vy };

    // DZZ: phi = norm*vz^2*t
    case 9:
      return { norm*(            neg2tg(vx)*vz*vz),
               norm*(            neg2tg(vy)*vz*vz),
               norm*(2.f*vz*t + neg2tg(vz)*vz*vz) };

    default: return {};
  }
}

// S-shell Hessian: second derivatives of phi = t.
// Matches g2g/cpu/functions.cpp hPX/hPY/hPZ/hIX/hIY/hIZ formulas for S.
//   px = d2phi/dx2 = vx^2*4*th - 2*tg
//   py = d2phi/dy2 = vy^2*4*th - 2*tg
//   pz = d2phi/dz2 = vz^2*4*th - 2*tg
//   ix = d2phi/dxdy = vx*vy*4*th
//   iy = d2phi/dxdz = vx*vz*4*th
//   iz = d2phi/dydz = vy*vz*4*th
inline RefHess ref_eval_gto_hess_S(float vx, float vy, float vz, float dist2,
                                    const std::vector<float>& alphas,
                                    const std::vector<float>& coeffs) {
  float t, tg, th;
  ref_gto_radial_tg_th(dist2, alphas, coeffs, t, tg, th);
  (void)t;
  float f4th = 4.f * th;
  return {
    vx*vx*f4th - 2.f*tg,  // px
    vy*vy*f4th - 2.f*tg,  // py
    vz*vz*f4th - 2.f*tg,  // pz
    vx*vy*f4th,            // ix = dxdy
    vx*vz*f4th,            // iy = dxdz
    vy*vz*f4th             // iz = dydz
  };
}

// ============================================================================
// Becke partition weights
// Matches g2g/cpu/weight.cpp and g2g/cuda/kernels/weight.h
// ============================================================================

// Inter-atomic Euclidean distance (double precision, matching weight.cpp).
inline double ref_atom_dist3(double ax, double ay, double az,
                               double bx, double by, double bz) {
  double dx = ax-bx, dy = ay-by, dz = az-bz;
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Becke step polynomial: s(u) = 1.5*u - 0.5*u^3, applied 3 times.
// Returns the cell-function value in [0,1].
inline double ref_becke_s3(double u) {
  u = 1.5*u - 0.5*(u*u*u);
  u = 1.5*u - 0.5*(u*u*u);
  u = 1.5*u - 0.5*(u*u*u);
  return 0.5*(1.0 - u);
}

// Becke weight for grid point (px,py,pz) belonging to atom `atom_of_point`.
//   atom_xyz:  [n_atoms*3] atom positions (x,y,z)
//   atom_rm:   [n_atoms]   van-der-Waals/covalent radii for heteronuclear adjustment
//   atom_dists:[n_atoms^2] precomputed pairwise inter-atomic distances (row-major)
//
// For homonuclear atoms (all rm equal) the heteronuclear adjustment vanishes.
inline double ref_cpu_becke_weight(const std::vector<double>& atom_xyz,
                                    const std::vector<double>& atom_rm,
                                    const std::vector<double>& atom_dists,
                                    double px, double py, double pz,
                                    int atom_of_point) {
  int n = (int)atom_rm.size();
  double P_total = 0.0;
  double P_atom  = 0.0;

  for (int j = 0; j < n; ++j) {
    double djx = atom_xyz[3*j], djy = atom_xyz[3*j+1], djz = atom_xyz[3*j+2];
    double d_Pj = ref_atom_dist3(px, py, pz, djx, djy, djz);
    double P_curr = 1.0;

    for (int k = 0; k < n; ++k) {
      if (k == j) continue;
      double dkx = atom_xyz[3*k], dky = atom_xyz[3*k+1], dkz = atom_xyz[3*k+2];
      double d_Pk = ref_atom_dist3(px, py, pz, dkx, dky, dkz);
      double d_jk = atom_dists[j * n + k];

      double u = (d_Pj - d_Pk) / d_jk;

      // Heteronuclear adjustment (Becke eq. A5)
      double x = atom_rm[j] / atom_rm[k];
      x = (x - 1.0) / (x + 1.0);
      if (std::abs(x) > 1e-10)
        u += (x / (x*x - 1.0)) * (1.0 - u*u);

      P_curr *= ref_becke_s3(u);
      if (P_curr == 0.0) break;
    }

    P_total += P_curr;
    if (j == atom_of_point) P_atom = P_curr;
  }

  return (P_total == 0.0) ? 0.0 : P_atom / P_total;
}

// Helper: build [n_atoms^2] pairwise distance table from [n_atoms*3] positions.
inline std::vector<double> ref_make_atom_dists(const std::vector<double>& xyz) {
  int n = (int)xyz.size() / 3;
  std::vector<double> d(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      d[i*n+j] = ref_atom_dist3(xyz[3*i],xyz[3*i+1],xyz[3*i+2],
                                  xyz[3*j],xyz[3*j+1],xyz[3*j+2]);
  return d;
}

// ============================================================================
// Electron density from density matrix and function values
// Matches g2g/cpu/iteration.cpp and g2g/cuda/kernels/energy.h (LDA branch)
// ============================================================================

// rho(p) = sum_i fv[m*p+i] * sum_{j<=i} rmm[i*m+j] * fv[m*p+j]
// rmm: lower-triangular density matrix [m*m] (upper triangle may be 0).
// fv:  function values, layout fv[m*point + func].
inline float ref_cpu_density(const std::vector<float>& rmm, int m,
                               const std::vector<float>& fv, int p) {
  float rho = 0.f;
  for (int i = 0; i < m; ++i) {
    float wi = 0.f;
    for (int j = 0; j <= i; ++j) wi += rmm[i * m + j] * fv[m * p + j];
    rho += fv[m * p + i] * wi;
  }
  return rho;
}

// ============================================================================
// Force integration
// Matches g2g/cuda/kernels/force.h (gpu_compute_forces)
// ============================================================================

// forces[a] = sum_{p=0}^{pts-1} density_deriv[COALESCED_DIM(pts)*a + p] * force_factors[p]
//
// F4T must have public float fields .x .y .z .w
// (works with RefF4 in CPU tests and G2G::vec_type<float,4> in CUDA tests).
template<typename F4T>
std::vector<F4T> ref_cpu_forces(int n_atoms, int pts,
                                  const std::vector<float>& factors,
                                  const std::vector<F4T>& derivs) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4T> out(n_atoms);
  for (int a = 0; a < n_atoms; ++a) {
    float ax = 0.f, ay = 0.f, az = 0.f, aw = 0.f;
    for (int p = 0; p < pts; ++p) {
      float f = factors[p];
      ax += derivs[cdim * a + p].x * f;
      ay += derivs[cdim * a + p].y * f;
      az += derivs[cdim * a + p].z * f;
      aw += derivs[cdim * a + p].w * f;
    }
    out[a].x = ax; out[a].y = ay; out[a].z = az; out[a].w = aw;
  }
  return out;
}

// ============================================================================
// Density derivatives from density matrix and gradient values
// Matches g2g/cuda/kernels/energy_derivs.h (gpu_compute_density_derivs)
// ============================================================================

// For each basis function i:
//   w_i = sum_k R[k][i] * fv[cdim*k+p] * (i==k ? 2 : 1)
//   density_deriv[COALESCED_DIM(pts)*nuc[i] + p] -= grad_phi_i * w_i
//
// RMM: R[k][i] = rmm[k*m+i] (lower-triangular convention; R[i][k]=R[k][i]).
// fv layout: fv[COALESCED_DIM(pts)*func + point]
// gv layout: gv[COALESCED_DIM(pts)*func + point]  (gradient of phi_i at point p)
// F4T must have public float fields .x .y .z .w
template<typename F4T>
std::vector<F4T> ref_cpu_density_derivs(
    const std::vector<float>& rmm, int m,
    const std::vector<float>& fv, const std::vector<F4T>& gv,
    const std::vector<unsigned>& nuc, int nuc_count, int pts) {
  int cdim = COALESCED_DIMENSION(pts);
  std::vector<F4T> deriv(cdim * nuc_count);
  for (auto& d : deriv) { d.x = 0.f; d.y = 0.f; d.z = 0.f; d.w = 0.f; }

  for (int p = 0; p < pts; ++p) {
    for (int i = 0; i < m; ++i) {
      float w = 0.f;
      for (int k = 0; k < m; ++k)
        w += rmm[k * m + i] * fv[cdim * k + p] * ((i == k) ? 2.f : 1.f);
      int ni = (int)nuc[i];
      deriv[cdim * ni + p].x -= gv[cdim * i + p].x * w;
      deriv[cdim * ni + p].y -= gv[cdim * i + p].y * w;
      deriv[cdim * ni + p].z -= gv[cdim * i + p].z * w;
      deriv[cdim * ni + p].w -= gv[cdim * i + p].w * w;
    }
  }
  return deriv;
}
