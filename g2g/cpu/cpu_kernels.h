#pragma once
// Standalone mathematical kernels extracted from g2g/cpu/weight.cpp,
// functions.cpp, and iteration.cpp.
//
// These are pure-math free functions with no dependency on FortranVars,
// HostMatrix, or any other G2G infrastructure.  They are callable from
// unit tests and from the refactored class methods.
//
// Covered algorithms (one extracted function per algorithm):
//   cpu_becke_weight_for_point() — Becke partitioning for one point (weight.cpp)
//   cpu_eval_gto_shell()         — GTO shell evaluation (functions.cpp)
//   cpu_compute_density_lda()    — LDA electron density for one point (iteration.cpp)

#include <cmath>
#include <vector>

namespace G2G {

// ============================================================================
// Becke partition weight for one integration point
// Extracted from PointGroupCPU::compute_weights() in weight.cpp.
// ============================================================================
//
// Computes:  P_atom / P_total  (or 0.0 when P_total == 0.0).
//
// Parameters:
//   px, py, pz       : integration point coordinates
//   home_atom        : global index of the atom this point belongs to
//   local_nuc        : array[n_nuc] of global atom indices for the local group
//   n_nuc            : number of local atoms
//   atom_x/y/z       : [total_atoms] global atom x/y/z coordinates
//   atom_rm          : [total_atoms] van der Waals radii for het.-nuclear adjust.
//   atom_dists       : [total_atoms * total_atoms] precomputed inter-atom distances
//                      stored row-major: atom_dists[j * total_atoms + k]
//   total_atoms      : size of atom_x/y/z/rm arrays; stride of atom_dists
//   home_in_local    : true if home_atom appears in local_nuc[]
//                      (false triggers the "punto sin atomo" special path)
//
// Matches weight.cpp exactly: heteronuclear Becke correction, 3× polynomial.
inline double cpu_becke_weight_for_point(
    double px, double py, double pz,
    unsigned home_atom,
    const unsigned* local_nuc, unsigned n_nuc,
    const double* atom_x, const double* atom_y, const double* atom_z,
    const double* atom_rm,
    const double* atom_dists, unsigned total_atoms,
    bool home_in_local) {

  // Distance from integration point to a global atom
  auto dist_to = [&](unsigned a) -> double {
    double dx = px - atom_x[a], dy = py - atom_y[a], dz = pz - atom_z[a];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };

  // Becke cell function: u → s(u) applied 3 times → 0.5*(1-u)
  // Matches the 3 inline applications in weight.cpp lines 44-47.
  auto becke_s3 = [](double u) -> double {
    u = 1.5 * u - 0.5 * (u * u * u);
    u = 1.5 * u - 0.5 * (u * u * u);
    u = 1.5 * u - 0.5 * (u * u * u);
    return 0.5 * (1.0 - u);
  };

  // One cell-function step: compute s(mu_jk) for a given pair of atoms.
  // Applies the heteronuclear adjustment (Becke eq. A5) then the 3× polynomial.
  // Matches weight.cpp inner k-loop body exactly.
  auto becke_mu = [&](unsigned aj, unsigned ak, double d_Paj, double d_Pak) -> double {
    double d_jk = atom_dists[aj * total_atoms + ak];
    double u = (d_Paj - d_Pak) / d_jk;
    double x = atom_rm[aj] / atom_rm[ak];
    x = (x - 1.0) / (x + 1.0);
    u += (x / (x * x - 1.0)) * (1.0 - u * u);
    return becke_s3(u);
  };

  double P_total = 0.0;
  double P_atom  = 0.0;

  // Main path: iterate over local atoms, accumulate P_total and P_atom.
  for (unsigned j = 0; j < n_nuc; ++j) {
    unsigned aj     = local_nuc[j];
    double   d_Paj  = dist_to(aj);
    double   P_curr = 1.0;

    for (unsigned k = 0; k < n_nuc; ++k) {
      unsigned ak = local_nuc[k];
      if (ak == aj) continue;
      P_curr *= becke_mu(aj, ak, d_Paj, dist_to(ak));
      if (P_curr == 0.0) break;
    }

    if (aj == home_atom) {
      P_atom = P_curr;
      if (P_atom == 0.0) break;
    }
    P_total += P_curr;
  }

  // Special path: home_atom not in local group ("punto sin atomo propio").
  // Matches weight.cpp lines 68-96: P_atom computed separately.
  if (!home_in_local) {
    P_atom        = 1.0;
    double d_Phome = dist_to(home_atom);
    double rm_home = atom_rm[home_atom];

    for (unsigned k = 0; k < n_nuc; ++k) {
      unsigned ak    = local_nuc[k];
      double   d_jk  = atom_dists[home_atom * total_atoms + ak];
      double   d_Pak = dist_to(ak);
      double   u     = (d_Phome - d_Pak) / d_jk;
      double   x     = rm_home / atom_rm[ak];
      x = (x - 1.0) / (x + 1.0);
      u += (x / (x * x - 1.0)) * (1.0 - u * u);
      P_atom *= becke_s3(u);
      if (P_atom == 0.0) break;
    }
  }

  return (P_total == 0.0) ? 0.0 : (P_atom / P_total);
}

// ============================================================================
// GTO shell evaluation at a displacement vector v = point - atom_position
// Extracted from PointGroupCPU::compute_functions() in functions.cpp.
// ============================================================================
//
// Evaluates one Gaussian shell (S / P / D) and optionally its gradient and
// Hessian.  The shell type selects the number of output functions nf:
//   shell_type 0 → S  (nf = 1): phi = t
//   shell_type 1 → P  (nf = 3): phi = (vx·t, vy·t, vz·t)
//   shell_type 2 → D  (nf = 6): phi = norm·(vx²·t, vy·vx·t, vy²·t,
//                                              vz·vx·t, vz·vy·t, vz²·t)
//
// Parameters:
//   vx, vy, vz   : v = point_position − atom_position
//   dist2        : vx² + vy² + vz²
//   alphas[]     : Gaussian exponents (nc contractions)
//   coeffs[]     : Gaussian coefficients (nc contractions)
//   nc           : number of contractions
//   shell_type   : 0=S, 1=P, 2=D
//   norm         : normalization factor for D-shell diagonal components (DXX/DYY/DZZ)
//   compute_grad : fill gx/gy/gz outputs (pass non-null pointers)
//   compute_hess : fill hpx/hpy/hpz/hix/hiy/hiz outputs (pass non-null pointers)
//
// Outputs (each array must have room for nf elements):
//   val         : function values
//   gx, gy, gz  : gradient components (only when compute_grad)
//   hpx,hpy,hpz : diagonal Hessian d²φ/d{x,y,z}² (only when compute_hess)
//   hix,hiy,hiz : cross Hessian d²φ/dxdy, d²φ/dxdz, d²φ/dydz (only when compute_hess)
//
// Returns: nf (number of functions written: 1, 3, or 6).
//
// Matches functions.cpp inner loop verbatim — same sign conventions, same
// normalization placement for each shell component.
template <class scalar_type>
int cpu_eval_gto_shell(
    scalar_type vx, scalar_type vy, scalar_type vz, scalar_type dist2,
    const scalar_type* alphas, const scalar_type* coeffs, int nc,
    int shell_type, scalar_type norm,
    bool compute_grad, bool compute_hess,
    scalar_type* val,
    scalar_type* gx,  scalar_type* gy,  scalar_type* gz,
    scalar_type* hpx, scalar_type* hpy, scalar_type* hpz,
    scalar_type* hix, scalar_type* hiy, scalar_type* hiz) {

  // Radial contraction — identical exponent cutoff to functions.cpp
  scalar_type t = 0, tg = 0, th = 0;
  for (int c = 0; c < nc; ++c) {
    scalar_type expon = alphas[c] * dist2;
    if (expon > static_cast<scalar_type>(70.0)) continue;
    scalar_type t0 = std::exp(-expon) * coeffs[c];
    t  += t0;
    if (compute_grad || compute_hess) tg += t0 * alphas[c];
    if (compute_hess)                 th += t0 * (alphas[c] * alphas[c]);
  }

  // vxxy / vyzz helpers matching functions.cpp lines 79-80:
  //   vxxy = (v.x, v.x, v.y)
  //   vyzz = (v.y, v.z, v.z)
  // Used for cross-Hessian: hI{X,Y,Z} = vxxy·vyzz * 4*th * ...
  const scalar_type vxxy_x = vx, vxxy_y = vx, vxxy_z = vy;
  const scalar_type vyzz_x = vy, vyzz_y = vz, vyzz_z = vz;

  if (shell_type == 0) {
    // ── S shell ─────────────────────────────────────────────────────────────
    val[0] = t;
    if (compute_grad) {
      gx[0] = vx * (-2 * tg);
      gy[0] = vy * (-2 * tg);
      gz[0] = vz * (-2 * tg);
    }
    if (compute_hess) {
      hpx[0] = vx * vx * 4 * th - 2 * tg;
      hpy[0] = vy * vy * 4 * th - 2 * tg;
      hpz[0] = vz * vz * 4 * th - 2 * tg;
      hix[0] = vxxy_x * vyzz_x * 4 * th;  // vx*vy
      hiy[0] = vxxy_y * vyzz_y * 4 * th;  // vx*vz
      hiz[0] = vxxy_z * vyzz_z * 4 * th;  // vy*vz
    }
    return 1;

  } else if (shell_type == 1) {
    // ── P shell ──────────────────────────────────────────────────────────────
    val[0] = vx * t;
    val[1] = vy * t;
    val[2] = vz * t;
    if (compute_grad) {
      gx[0] = t - vx * 2 * tg * vx;
      gy[0] =   - vy * 2 * tg * vx;
      gz[0] =   - vz * 2 * tg * vx;
      gx[1] =   - vx * 2 * tg * vy;
      gy[1] = t - vy * 2 * tg * vy;
      gz[1] =   - vz * 2 * tg * vy;
      gx[2] =   - vx * 2 * tg * vz;
      gy[2] =   - vy * 2 * tg * vz;
      gz[2] = t - vz * 2 * tg * vz;
    }
    if (compute_hess) {
      // P Hessians from functions.cpp lines 130-157
      hpx[0] = vx*vx*4*th*vx - 6*tg*vx;
      hpy[0] = vy*vy*4*th*vx - 2*tg*vx;
      hpz[0] = vz*vz*4*th*vx - 2*tg*vx;
      hix[0] = vxxy_x*vyzz_x*4*th*vx - vy*2*tg;
      hiy[0] = vxxy_y*vyzz_y*4*th*vx - vz*2*tg;
      hiz[0] = vxxy_z*vyzz_z*4*th*vx;
      hpx[1] = vx*vx*4*th*vy - 2*tg*vy;
      hpy[1] = vy*vy*4*th*vy - 6*tg*vy;
      hpz[1] = vz*vz*4*th*vy - 2*tg*vy;
      hix[1] = vxxy_x*vyzz_x*4*th*vy - 2*tg*vx;
      hiy[1] = vxxy_y*vyzz_y*4*th*vy;
      hiz[1] = vxxy_z*vyzz_z*4*th*vy - 2*tg*vz;
      hpx[2] = vx*vx*4*th*vz - 2*tg*vz;
      hpy[2] = vy*vy*4*th*vz - 2*tg*vz;
      hpz[2] = vz*vz*4*th*vz - 6*tg*vz;
      hix[2] = vxxy_x*vyzz_x*4*th*vz;
      hiy[2] = vxxy_y*vyzz_y*4*th*vz - 2*tg*vx;
      hiz[2] = vxxy_z*vyzz_z*4*th*vz - 2*tg*vy;
    }
    return 3;

  } else {
    // ── D shell ──────────────────────────────────────────────────────────────
    // Components: [0]=DXX [1]=DXY [2]=DYY [3]=DXZ [4]=DYZ [5]=DZZ
    val[0] = t * vx * vx * norm;
    val[1] = t * vy * vx;
    val[2] = t * vy * vy * norm;
    val[3] = t * vz * vx;
    val[4] = t * vz * vy;
    val[5] = t * vz * vz * norm;
    if (compute_grad) {
      gx[0] = (2*vx*t - vx*2*tg*vx*vx) * norm;
      gy[0] = (      - vy*2*tg*vx*vx)  * norm;
      gz[0] = (      - vz*2*tg*vx*vx)  * norm;
      gx[1] = vy*t - vx*2*tg*vy*vx;
      gy[1] = vx*t - vy*2*tg*vy*vx;
      gz[1] =      - vz*2*tg*vy*vx;
      gx[2] = (      - vx*2*tg*vy*vy)  * norm;
      gy[2] = (2*vy*t - vy*2*tg*vy*vy) * norm;
      gz[2] = (      - vz*2*tg*vy*vy)  * norm;
      gx[3] = vz*t - vx*2*tg*vz*vx;
      gy[3] =      - vy*2*tg*vz*vx;
      gz[3] = vx*t - vz*2*tg*vz*vx;
      gx[4] =      - vx*2*tg*vz*vy;
      gy[4] = vz*t - vy*2*tg*vz*vy;
      gz[4] = vy*t - vz*2*tg*vz*vy;
      gx[5] = (      - vx*2*tg*vz*vz)  * norm;
      gy[5] = (      - vy*2*tg*vz*vz)  * norm;
      gz[5] = (2*vz*t - vz*2*tg*vz*vz) * norm;
    }
    if (compute_hess) {
      // D Hessians from functions.cpp lines 222-351
      hpx[0] = (vx*vx*4*th*vx*vx - 10*tg*vx*vx + 2*t) * norm;
      hpy[0] = (vy*vy*4*th*vx*vx -  2*tg*vx*vx     )  * norm;
      hpz[0] = (vz*vz*4*th*vx*vx -  2*tg*vx*vx     )  * norm;
      hix[0] = (vxxy_x*vyzz_x*4*th*vx*vx - 4*tg*vxxy_x*vyzz_x) * norm;
      hiy[0] = (vxxy_y*vyzz_y*4*th*vx*vx - 4*tg*vxxy_y*vyzz_y) * norm;
      hiz[0] = (vxxy_z*vyzz_z*4*th*vx*vx                      ) * norm;

      hpx[1] = vx*vx*4*th*vx*vy - 6*tg*vx*vy;
      hpy[1] = vy*vy*4*th*vx*vy - 6*tg*vx*vy;
      hpz[1] = vz*vz*4*th*vx*vy - 2*tg*vx*vy;
      hix[1] = vxxy_x*vyzz_x*4*th*vx*vy - 2*(vx*vx+vy*vy)*tg + t;
      hiy[1] = vxxy_y*vyzz_y*4*th*vx*vy - 2*vy*vz*tg;
      hiz[1] = vxxy_z*vyzz_z*4*th*vx*vy - 2*vx*vz*tg;

      hpx[2] = (vx*vx*4*th*vy*vy -  2*tg*vy*vy     )  * norm;
      hpy[2] = (vy*vy*4*th*vy*vy - 10*tg*vy*vy + 2*t) * norm;
      hpz[2] = (vz*vz*4*th*vy*vy -  2*tg*vy*vy     )  * norm;
      hix[2] = (vxxy_x*vyzz_x*4*th*vy*vy - 4*tg*vxxy_x*vyzz_x) * norm;
      hiy[2] = (vxxy_y*vyzz_y*4*th*vy*vy                      ) * norm;
      hiz[2] = (vxxy_z*vyzz_z*4*th*vy*vy - 4*tg*vxxy_z*vyzz_z) * norm;

      hpx[3] = vx*vx*4*th*vx*vz - 6*tg*vx*vz;
      hpy[3] = vy*vy*4*th*vx*vz - 2*tg*vx*vz;
      hpz[3] = vz*vz*4*th*vx*vz - 6*tg*vx*vz;
      hix[3] = vxxy_x*vyzz_x*4*th*vx*vz - 2*vy*vz*tg;
      hiy[3] = vxxy_y*vyzz_y*4*th*vx*vz - 2*(vx*vx+vz*vz)*tg + t;
      hiz[3] = vxxy_z*vyzz_z*4*th*vx*vz - 2*vx*vy*tg;

      hpx[4] = vx*vx*4*th*vy*vz - 2*tg*vy*vz;
      hpy[4] = vy*vy*4*th*vy*vz - 6*tg*vy*vz;
      hpz[4] = vz*vz*4*th*vy*vz - 6*tg*vy*vz;
      hix[4] = vxxy_x*vyzz_x*4*th*vy*vz - 2*vx*vz*tg;
      hiy[4] = vxxy_y*vyzz_y*4*th*vy*vz - 2*vx*vy*tg;
      hiz[4] = vxxy_z*vyzz_z*4*th*vy*vz - 2*(vy*vy+vz*vz)*tg + t;

      hpx[5] = (vx*vx*4*th*vz*vz -  2*tg*vz*vz     )  * norm;
      hpy[5] = (vy*vy*4*th*vz*vz -  2*tg*vz*vz     )  * norm;
      hpz[5] = (vz*vz*4*th*vz*vz - 10*tg*vz*vz + 2*t) * norm;
      hix[5] = (vxxy_x*vyzz_x*4*th*vz*vz                      ) * norm;
      hiy[5] = (vxxy_y*vyzz_y*4*th*vz*vz - 4*tg*vxxy_y*vyzz_y) * norm;
      hiz[5] = (vxxy_z*vyzz_z*4*th*vz*vz - 4*tg*vxxy_z*vyzz_z) * norm;
    }
    return 6;
  }
}

// ============================================================================
// LDA electron density for one integration point
// Extracted from PointGroupCPU::solve_closed() LDA branch in iteration.cpp.
// ============================================================================
//
// Computes:  rho = sum_i F_i * sum_{j >= i} rmm[i*m + j] * F_j
//
// Parameters:
//   fv[m]       : function values at this point (fv[func])
//   rmm[m * m]  : FULL SYMMETRIC density matrix, row-major rmm[row*m + col]
//                 Both triangles must be filled with the same physical values
//                 (as produced by PointGroupCPU::get_rmm_input()).
//   m           : number of basis functions in this group
//
// Returns:  partial electron density (scalar_type)
//
// Note: this sums the UPPER triangle (j >= i) of the symmetric matrix, which
// is equivalent to the lower-triangle reference sum because rmm is symmetric.
// Using the upper triangle matches the iteration.cpp LDA loop exactly.
template <class scalar_type>
scalar_type cpu_compute_density_lda(const scalar_type* fv,
                                     const scalar_type* rmm, int m,
                                     int rmm_stride = 0) {
  if (rmm_stride == 0) rmm_stride = m;
  scalar_type rho = 0;
  for (int i = 0; i < m; ++i) {
    scalar_type w = 0;
    // Matches iteration.cpp: for (int j = i; j < group_m; j++)
    //                            w += rmm_input(j, i) * Fj;
    // rmm_input(j, i) = data[i * rmm_stride + j] in our flat layout.
    for (int j = i; j < m; ++j)
      w += rmm[i * rmm_stride + j] * fv[j];
    rho += fv[i] * w;
  }
  return rho;
}

// ============================================================================
// GGA electron density and gradient sums for one integration point
// Extracted from PointGroupCPU::solve_closed() GGA branch in iteration.cpp.
// ============================================================================
//
// For each i, accumulates contributions from j <= i (LOWER TRIANGLE of rmm):
//   w     = sum_{j<=i} rmm[i*m+j] * fv[j]        (density weight)
//   w3x/y/z = sum_{j<=i} rmm[i*m+j] * gx/y/z[j]  (gradient weights)
//   ww1x/y/z = sum_{j<=i} rmm[i*m+j] * hpx/y/z[j] (diagonal Hessian weights)
//   ww2x/y/z = sum_{j<=i} rmm[i*m+j] * hix/y/z[j] (cross Hessian weights)
//
// Then accumulates into the output struct:
//   pd   += fv[i] * w
//   tdx  += gx[i]*w + w3x*fv[i]          (d(rho)/dx)
//   tdd1x += 2*gx[i]*w3x + hpx[i]*w + ww1x*fv[i]  (d2(rho)/dx2)
//   tdd2x += gx[i]*w3y + gy[i]*w3x + hix[i]*w + ww2x*fv[i]  (d2(rho)/dxdy)
//   (similarly for y, z components)
//
// Parameters:
//   fv[m]                    : basis function values at this point
//   gxv/gyv/gzv[m]           : gradient components of basis functions
//   hpxv/hpyv/hpzv[m]        : diagonal Hessian d2phi/dx2, dy2, dz2
//   hixv/hiyv/hizv[m]        : cross Hessian d2phi/dxdy, dxdz, dydz
//   rmm[m*m]                 : density matrix, row-major; lower triangle used
//                              (rmm[i*m+j] for j <= i; upper triangle ignored)
//   m                        : number of basis functions in this group
//
// Matches iteration.cpp solve_closed() GGA branch (the j <= i loop) exactly.
template <class scalar_type>
struct GGADensity {
  scalar_type pd;
  scalar_type tdx, tdy, tdz;
  scalar_type tdd1x, tdd1y, tdd1z;
  scalar_type tdd2x, tdd2y, tdd2z;
};

template <typename scalar_type>
GGADensity<scalar_type> cpu_compute_density_gga(
    const scalar_type* __restrict__ fv,
    const scalar_type* __restrict__ gxv,  const scalar_type* __restrict__ gyv,  const scalar_type* __restrict__ gzv,
    const scalar_type* __restrict__ hpxv, const scalar_type* __restrict__ hpyv, const scalar_type* __restrict__ hpzv,
    const scalar_type* __restrict__ hixv, const scalar_type* __restrict__ hiyv, const scalar_type* __restrict__ hizv,
    const scalar_type* __restrict__ rmm, int m, int rmm_stride = 0) {

    if (rmm_stride == 0) rmm_stride = m;
    GGADensity<scalar_type> res{};

    for (int i = 0; i < m; ++i) {
        scalar_type w = 0, w3xc = 0, w3yc = 0, w3zc = 0;
        scalar_type ww1xc = 0, ww1yc = 0, ww1zc = 0;
        scalar_type ww2xc = 0, ww2yc = 0, ww2zc = 0;

        const scalar_type* __restrict__ rmm_row = &rmm[i * rmm_stride];

        // Fission 1: Primary density and first-order gradients
        // #pragma GCC ivdep tells the compiler "ignore vector dependencies", 
        // allowing it to use its native auto-vectorizer instead of OpenMP's rigid SIMD rules.
        #pragma GCC ivdep
        for (int j = 0; j <= i; ++j) {
            scalar_type rmj = rmm_row[j];
            w    += fv[j]  * rmj;
            w3xc += gxv[j] * rmj;
            w3yc += gyv[j] * rmj;
            w3zc += gzv[j] * rmj;
        }

        // Anti-Fusion Barrier: This invisible inline assembly prevents GCC's optimizer 
        // from re-merging the loops and recreating the Register Pressure issue.
        asm volatile("" ::: "memory");

        // Fission 2: High-order partials (Set 1)
        #pragma GCC ivdep
        for (int j = 0; j <= i; ++j) {
            scalar_type rmj = rmm_row[j];
            ww1xc += hpxv[j] * rmj;
            ww1yc += hpyv[j] * rmj;
            ww1zc += hpzv[j] * rmj;
        }

        asm volatile("" ::: "memory");

        // Fission 3: High-order partials (Set 2)
        #pragma GCC ivdep
        for (int j = 0; j <= i; ++j) {
            scalar_type rmj = rmm_row[j];
            ww2xc += hixv[j] * rmj;
            ww2yc += hiyv[j] * rmj;
            ww2zc += hizv[j] * rmj;
        }

        // Final Scalar Reductions
        scalar_type Fi  = fv[i];
        scalar_type gx  = gxv[i],  gy  = gyv[i],  gz  = gzv[i];
        scalar_type hpx = hpxv[i], hpy = hpyv[i], hpz = hpzv[i];
        scalar_type hix = hixv[i], hiy = hiyv[i], hiz = hizv[i];

        res.pd     += Fi * w;
        res.tdx    += gx * w  + w3xc * Fi;
        res.tdy    += gy * w  + w3yc * Fi;
        res.tdz    += gz * w  + w3zc * Fi;
        res.tdd1x  += gx * w3xc * 2 + hpx * w + ww1xc * Fi;
        res.tdd1y  += gy * w3yc * 2 + hpy * w + ww1yc * Fi;
        res.tdd1z  += gz * w3zc * 2 + hpz * w + ww1zc * Fi;
        res.tdd2x  += gx * w3yc + gy * w3xc + hix * w + ww2xc * Fi;
        res.tdd2y  += gx * w3zc + gz * w3xc + hiy * w + ww2yc * Fi;
        res.tdd2z  += gy * w3zc + gz * w3yc + hiz * w + ww2zc * Fi;
    }
    return res;
}

// ============================================================================
// Batched GGA density + gradients for ALL points in a group at once.
// ============================================================================
//
// Computes, for every integration point p, exactly the same quantities as
// cpu_compute_density_gga() above (pd and the 9 gradient/hessian components),
// but vectorized over the *points* dimension instead of doing one scalar
// reduction per point.
//
// Why this is faster: the per-point kernel's inner loop is a float reduction
// (`w += fv[j]*rmm`), which the compiler cannot auto-vectorize without
// -ffast-math (reassociation is not value-safe). Here we transpose the input
// arrays to [function x point] layout so the inner accumulation runs over the
// independent point index — a pure axpy with no cross-iteration dependency,
// which -O3 -march=native vectorizes (AVX2+FMA) with no reassociation.
//
// Why it is bit-exact: each individual point p accumulates its w_k vectors over
// j in the same order (0..i) and its pd/grad outputs over i in the same order
// (0..m) as the scalar kernel. Vectorization only runs distinct points in
// distinct SIMD lanes; it never reorders a single point's summation.
//
// Inputs (point-major, element (p, j) = arr[p*src_stride + j]) match the
// HostMatrix arrays produced by compute_functions(): fv, gradients gx/gy/gz,
// hessian-diagonal hpx/hpy/hpz, hessian-offdiag hix/hiy/hiz.
//   rmm[m*m]      : symmetric density submatrix (off-diagonals pre-doubled),
//                   row-major rmm[i*rmm_stride + j]
//   m             : number of basis functions in the group
//   np            : number of integration points in the group
// Outputs (each length np, caller-allocated): pd and the 9 gradient terms,
// indexed by point.
template <typename scalar_type>
void cpu_compute_density_gga_batch(
    const scalar_type* __restrict__ fv,
    const scalar_type* __restrict__ gx,  const scalar_type* __restrict__ gy,  const scalar_type* __restrict__ gz,
    const scalar_type* __restrict__ hpx, const scalar_type* __restrict__ hpy, const scalar_type* __restrict__ hpz,
    const scalar_type* __restrict__ hix, const scalar_type* __restrict__ hiy, const scalar_type* __restrict__ hiz,
    const scalar_type* __restrict__ rmm, int m, int rmm_stride, int np, int src_stride,
    scalar_type* __restrict__ pd,
    scalar_type* __restrict__ tdx,   scalar_type* __restrict__ tdy,   scalar_type* __restrict__ tdz,
    scalar_type* __restrict__ tdd1x, scalar_type* __restrict__ tdd1y, scalar_type* __restrict__ tdd1z,
    scalar_type* __restrict__ tdd2x, scalar_type* __restrict__ tdd2y, scalar_type* __restrict__ tdd2z) {

  if (rmm_stride == 0) rmm_stride = m;

  // Point-tiling: process B points at a time. Within a tile, rmm (the shared
  // operand) is read once and stays L1/L2-resident across all i, while the
  // tile's transposed function data (10 * m * B) is small enough to fit L1.
  // This keeps the kernel compute-bound (as the scalar per-point version was)
  // while exposing the points dimension to the vectorizer as an axpy.
  constexpr int B = 8;  // one AVX-256 float vector (or 2 doubles); no remainder
  constexpr int NCH = 10;       // function channels: value + 3 grad + 6 hessian
  constexpr int FS = NCH * B;   // per-function stride in the interleaved tile

  // Interleaved transposed tile: T[(j*NCH + k)*B + b] holds channel k of
  // function j at lane b. Interleaving the 10 channels (rather than 10 separate
  // [m*B] arrays) makes the per-channel offset a *compile-time* k*B instead of a
  // runtime k*(m*B): the matvec inner loop below then addresses all 10 channels
  // off ONE base register (Tj) with constant displacements. With the old layout
  // the 10 runtime base pointers spilled to the stack and were reloaded every
  // j-iteration (1 stack-load per FMA — the loop was load-port bound). Same
  // arithmetic and summation order ⇒ bit-exact with the per-point kernel.
  static thread_local std::vector<scalar_type> tbuf;  // NCH * m * B tile
  const size_t tneed = (size_t)NCH * m * B;
  if (tbuf.size() < tneed) tbuf.resize(tneed);
  scalar_type* const T = tbuf.data();
  const scalar_type* const src[NCH] = {fv, gx, gy, gz, hpx, hpy, hpz, hix, hiy, hiz};

  for (int p0 = 0; p0 < np; p0 += B) {
    const int bn = (np - p0 < B) ? (np - p0) : B;

    // Transpose this tile: T[(j*NCH + k)*B + b] = src[k][(p0+b)*src_stride + j].
    for (int k = 0; k < NCH; ++k) {
      const scalar_type* __restrict__ s = src[k];
      for (int b = 0; b < bn; ++b) {
        const scalar_type* __restrict__ srow = s + (size_t)(p0 + b) * src_stride;
        for (int j = 0; j < m; ++j) T[(size_t)(j * NCH + k) * B + b] = srow[j];
      }
      for (int b = bn; b < B; ++b)
        for (int j = 0; j < m; ++j) T[(size_t)(j * NCH + k) * B + b] = scalar_type(0);
    }

    scalar_type acc_pd[B], acc_tdx[B], acc_tdy[B], acc_tdz[B];
    scalar_type acc_d1x[B], acc_d1y[B], acc_d1z[B];
    scalar_type acc_d2x[B], acc_d2y[B], acc_d2z[B];
    for (int b = 0; b < B; ++b) {
      acc_pd[b] = acc_tdx[b] = acc_tdy[b] = acc_tdz[b] = scalar_type(0);
      acc_d1x[b] = acc_d1y[b] = acc_d1z[b] = scalar_type(0);
      acc_d2x[b] = acc_d2y[b] = acc_d2z[b] = scalar_type(0);
    }

    for (int i = 0; i < m; ++i) {
      scalar_type Wfv[B], Wgx[B], Wgy[B], Wgz[B];
      scalar_type Whpx[B], Whpy[B], Whpz[B], Whix[B], Whiy[B], Whiz[B];
      for (int b = 0; b < B; ++b) {
        Wfv[b] = Wgx[b] = Wgy[b] = Wgz[b] = scalar_type(0);
        Whpx[b] = Whpy[b] = Whpz[b] = scalar_type(0);
        Whix[b] = Whiy[b] = Whiz[b] = scalar_type(0);
      }
      const scalar_type* __restrict__ rmm_row = &rmm[(size_t)i * rmm_stride];
      for (int j = 0; j <= i; ++j) {
        const scalar_type rmj = rmm_row[j];
        const scalar_type* __restrict__ Tj = T + (size_t)j * FS;
        for (int b = 0; b < B; ++b) {
          Wfv[b]  += Tj[0 * B + b] * rmj;
          Wgx[b]  += Tj[1 * B + b] * rmj;
          Wgy[b]  += Tj[2 * B + b] * rmj;
          Wgz[b]  += Tj[3 * B + b] * rmj;
          Whpx[b] += Tj[4 * B + b] * rmj;
          Whpy[b] += Tj[5 * B + b] * rmj;
          Whpz[b] += Tj[6 * B + b] * rmj;
          Whix[b] += Tj[7 * B + b] * rmj;
          Whiy[b] += Tj[8 * B + b] * rmj;
          Whiz[b] += Tj[9 * B + b] * rmj;
        }
      }
      const scalar_type* __restrict__ Ti = T + (size_t)i * FS;
      for (int b = 0; b < B; ++b) {
        const scalar_type Fi  = Ti[0 * B + b];
        const scalar_type igx = Ti[1 * B + b],  igy = Ti[2 * B + b],  igz = Ti[3 * B + b];
        const scalar_type ihpx = Ti[4 * B + b], ihpy = Ti[5 * B + b], ihpz = Ti[6 * B + b];
        const scalar_type ihix = Ti[7 * B + b], ihiy = Ti[8 * B + b], ihiz = Ti[9 * B + b];
        const scalar_type w = Wfv[b];
        acc_pd[b]  += Fi * w;
        acc_tdx[b] += igx * w + Wgx[b] * Fi;
        acc_tdy[b] += igy * w + Wgy[b] * Fi;
        acc_tdz[b] += igz * w + Wgz[b] * Fi;
        acc_d1x[b] += igx * Wgx[b] * 2 + ihpx * w + Whpx[b] * Fi;
        acc_d1y[b] += igy * Wgy[b] * 2 + ihpy * w + Whpy[b] * Fi;
        acc_d1z[b] += igz * Wgz[b] * 2 + ihpz * w + Whpz[b] * Fi;
        acc_d2x[b] += igx * Wgy[b] + igy * Wgx[b] + ihix * w + Whix[b] * Fi;
        acc_d2y[b] += igx * Wgz[b] + igz * Wgx[b] + ihiy * w + Whiy[b] * Fi;
        acc_d2z[b] += igy * Wgz[b] + igz * Wgy[b] + ihiz * w + Whiz[b] * Fi;
      }
    }

    for (int b = 0; b < bn; ++b) {
      pd[p0 + b]    = acc_pd[b];
      tdx[p0 + b]   = acc_tdx[b];
      tdy[p0 + b]   = acc_tdy[b];
      tdz[p0 + b]   = acc_tdz[b];
      tdd1x[p0 + b] = acc_d1x[b];
      tdd1y[p0 + b] = acc_d1y[b];
      tdd1z[p0 + b] = acc_d1z[b];
      tdd2x[p0 + b] = acc_d2x[b];
      tdd2y[p0 + b] = acc_d2y[b];
      tdd2z[p0 + b] = acc_d2z[b];
    }
  }
}

// ============================================================================
// Force density derivatives for one integration point (closed-shell)
// Extracted from PointGroupCPU::solve_closed() force loop in iteration.cpp.
// ============================================================================
//
// For each flat basis function ii:
//   w_ii = sum_j rmm[ii*m+j] * fv[j] * (ii==j ? 2 : 1)
//   ddx[func2nuc[ii]] -= w_ii * gxv[ii]    (additive into output arrays)
//
// Parameters:
//   fv[m]          : function values at this point
//   gxv/gyv/gzv[m] : gradient components at this point
//   rmm[m*m]       : FULL SYMMETRIC density matrix, row-major rmm[ii*m+j]
//                    (as produced by get_rmm_input — both triangles filled)
//   m              : number of basis functions
//   func2nuc[m]    : local atom index for each flat basis function
//   n_atoms        : number of local atoms (size of ddx/ddy/ddz arrays)
//   ddx/ddy/ddz    : [n_atoms] force contribution arrays (ADDITIVE — caller zeroes)
//
// Matches the inner per-function loop of solve_closed() and solve_opened()
// force sections exactly.
template <class scalar_type>
void cpu_compute_density_derivs(
    const scalar_type* fv,
    const scalar_type* gxv, const scalar_type* gyv, const scalar_type* gzv,
    const scalar_type* rmm, uint m,
    const unsigned* func2nuc, uint n_atoms,
    scalar_type* ddx, scalar_type* ddy, scalar_type* ddz,
    int rmm_stride = 0) {
  if (rmm_stride == 0) rmm_stride = (int)m;
  for (int ii = 0; ii < (int)m; ++ii) {
    scalar_type w = 0;
    for (int j = 0; j < (int)m; ++j)
      w += rmm[ii * rmm_stride + j] * fv[j] * (ii == j ? 2 : 1);
    int nuc = (int)func2nuc[ii];
    ddx[nuc] -= w * gxv[ii];
    ddy[nuc] -= w * gyv[ii];
    ddz[nuc] -= w * gzv[ii];
  }
}

// ============================================================================
// RMM element update: weighted dot product of two transposed function rows
// Extracted from PointGroupCPU::solve_closed() RMM section in iteration.cpp.
// ============================================================================
//
// Computes:  sum_{p=0}^{npoints-1} fv_row[p] * fv_col[p] * factors[p]
//
// Parameters:
//   fv_row[npoints]  : transposed function values for the row basis function
//   fv_col[npoints]  : transposed function values for the col basis function
//   factors[npoints] : per-point weight factors (= point_weight * y2a)
//   npoints          : number of integration points
//
// Returns: the dot product (scalar_type).  Caller casts to double before
// accumulating into rmm_global_output (which is always double-precision).
//
// Matches the inner point loop of the RMM section in solve_closed/solve_opened
// exactly, except that precision of the accumulation follows scalar_type.
template <class scalar_type>
scalar_type cpu_update_rmm(const scalar_type* __restrict__ fv_row, 
                            const scalar_type* __restrict__ fv_col,
                            const scalar_type* __restrict__ factors, int npoints) {
  scalar_type res = 0;
  #pragma omp simd reduction(+:res)
  for (int p = 0; p < npoints; ++p) {
    res += fv_row[p] * fv_col[p] * factors[p];
  }
  return res;
}
}  // namespace G2G
