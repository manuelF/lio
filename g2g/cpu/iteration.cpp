#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <omp.h>
#include <cstdio>
#include "../common.h"
#include "../init.h"
#include "../cuda_includes.h"
#include "../matrix.h"
#include "../timer.h"
#include "../partition.h"

#include <stdlib.h>
#include "cpu_kernels.h"
#include "../pointxc/calc_ggaCS.h"
#include "../pointxc/calc_ggaOS.h"
#include "../pointxc/calc_ldaCS.h"

#if USE_LIBXC
#include <xc.h>
#include "../libxc/libxcproxy.h"
#endif

using std::cout;
using std::endl;
using std::vector;

namespace G2G {

template <class scalar_type>
void PointGroupCPU<scalar_type>::solve(
    Timers& timers, bool compute_rmm, bool lda, bool compute_forces,
    bool compute_energy, double& energy, double& energy_i, double& energy_c,
    double& energy_c1, double& energy_c2, HostMatrix<double>& fort_forces,
    int inner_threads, HostMatrix<double>& rmm_global_output, bool OPEN) {}

template <class scalar_type>
void PointGroupCPU<scalar_type>::solve_closed(
    Timers& timers, bool compute_rmm, bool lda, bool compute_forces,
    bool compute_energy, double& energy, HostMatrix<double>& fort_forces,
    int inner_threads, HostMatrix<double>& rmm_global_output,
    HostMatrix<double>& becke_dens, CDFTVars& my_cdft_vars) {
  const uint group_m = this->total_functions();
  const int npoints = this->points.size();

#if CPU_RECOMPUTE or !GPU_KERNELS
  /** Compute functions **/
  timers.functions.start();
  compute_functions(compute_forces, !lda);
  timers.functions.pause();
#endif

#if USE_LIBXC

#define libxc_init_param \
  fortran_vars.func_id, fortran_vars.func_coef, fortran_vars.nx_func, \
  fortran_vars.nc_func, fortran_vars.nsr_id, fortran_vars.screen, \
  XC_UNPOLARIZED
  LibxcProxy<scalar_type,3> libxcProxy(libxc_init_param);
#undef libxc_init_param

#endif

  double localenergy = 0.0;

  // prepare rmm_input for this group
  timers.density.start();

  HostMatrix<scalar_type> rmm_input(group_m, group_m);
  get_rmm_input(rmm_input);

  vector<vec_type3> forces;
  vector<std::vector<vec_type3> > forces_mat;
  HostMatrix<scalar_type> factors_rmm;
  HostMatrix<scalar_type> factors_cdft;

  if (compute_rmm || compute_forces) {
    factors_rmm.resize(this->points.size(), 1);
    if (my_cdft_vars.do_chrg) {
      factors_cdft.resize(this->points.size(), my_cdft_vars.regions);
      factors_cdft.zero();
    }
  }

  if (compute_forces) {
    forces.resize(this->total_nucleii(), vec_type3(0.f, 0.f, 0.f));
    forces_mat.resize(
        this->points.size(),
        vector<vec_type3>(this->total_nucleii(), vec_type3(0.f, 0.f, 0.f)));
  }

  const int iexch = fortran_vars.iexch;

  /** density **/
  if (lda) {
    for (int point = 0; point < (int)this->points.size(); point++) {
      // cpu_compute_density_lda() matches the original upper-triangle loop:
      //   sum_i Fi * sum_{j>=i} rmm_input(j,i) * Fj
      // rmm_input is symmetric (both triangles filled by get_rmm_input).
      scalar_type partial_density = cpu_compute_density_lda(
          function_values.row(point), rmm_input.asArray(), group_m,
          rmm_input.stride);

      scalar_type exc = 0.0, corr = 0.0, y2a = 0.0;

      calc_ldaCS_in(partial_density, exc, corr, y2a, iexch);

      if (compute_energy) {
        localenergy +=
            (partial_density * this->points[point].weight) * (exc + corr);
      }

      /** RMM **/
      if (compute_rmm || compute_forces) {
        factors_rmm(point) = this->points[point].weight * y2a;
      }
    }
  } else {
#pragma omp parallel for num_threads(inner_threads) \
    reduction(+ : localenergy) schedule(static)
    for (int point = 0; point < npoints; point++) {
      GGADensity<scalar_type> d = cpu_compute_density_gga(
          function_values.row(point),
          gX.row(point), gY.row(point), gZ.row(point),
          hPX.row(point), hPY.row(point), hPZ.row(point),
          hIX.row(point), hIY.row(point), hIZ.row(point),
          rmm_input.asArray(), group_m, rmm_input.stride);

      /** energy / potential **/
      scalar_type exc = 0.0, corr = 0.0, y2a = 0.0;
      const vec_type3 dxyz(d.tdx, d.tdy, d.tdz);
      const vec_type3 dd1(d.tdd1x, d.tdd1y, d.tdd1z);
      const vec_type3 dd2(d.tdd2x, d.tdd2y, d.tdd2z);
      scalar_type pd = d.pd;

#if USE_LIBXC
    /** Libxc CPU - version **/
    libxcProxy.doSCF(pd,dxyz,dd1,dd2,exc,corr,y2a);
#else
    calc_ggaCS_in<scalar_type, 3>(pd, dxyz, dd1, dd2, exc, corr, y2a, iexch,
                                  fortran_vars.fexc);
#endif

      const scalar_type wp = this->points[point].weight;

      if (compute_energy) {
        localenergy += (pd * wp) * (exc + corr);

        // Also calculates Becke partition if needed.
        if (fortran_vars.becke) {
          for (int i = 0; i < fortran_vars.atoms; i++) {
            becke_dens(i) += wp * pd * (this->points[point].atom_weights(i));
          }
        }
      }

      /** RMM **/
      if (compute_rmm || compute_forces) {
        factors_rmm(point) = wp * y2a;

        // Factors for CDFT.
        if (my_cdft_vars.do_chrg) {
          for (int i = 0; i < my_cdft_vars.regions; i++) {
            for (int j = 0; j < my_cdft_vars.natom(i); j++) {
              factors_cdft(point,i) = wp
                                    * (this->points[point].atom_weights(my_cdft_vars.atoms(j,i)));
            }
          }
        }
      }
    }
  }
  timers.density.pause();

  timers.forces.start();
  if (compute_forces) {
    // Build flat per-function atom index once from the shell structure.
    vector<unsigned> func2nuc_vec(group_m);
    for (int i = 0, ii = 0; i < (int)this->total_functions_simple(); i++) {
      unsigned nuc = this->func2local_nuc(ii);
      uint inc = this->small_function_type(i);
      for (uint k = 0; k < inc; k++, ii++) func2nuc_vec[ii] = nuc;
    }

    HostMatrix<scalar_type> ddx, ddy, ddz;
    ddx.resize(this->total_nucleii(), 1);
    ddy.resize(this->total_nucleii(), 1);
    ddz.resize(this->total_nucleii(), 1);
#pragma omp parallel for num_threads(inner_threads)
    for (int point = 0; point < (int)this->points.size(); point++) {
      ddx.zero(); ddy.zero(); ddz.zero();
      cpu_compute_density_derivs(
          function_values.row(point),
          gX.row(point), gY.row(point), gZ.row(point),
          rmm_input.asArray(), group_m,
          func2nuc_vec.data(), this->total_nucleii(),
          ddx.data, ddy.data, ddz.data,
          rmm_input.stride);
      scalar_type factor = factors_rmm(point);
      for (int i = 0; i < (int)this->total_nucleii(); i++) {
        forces_mat[point][i] = vec_type3(ddx(i), ddy(i), ddz(i)) * factor;
      }
    }
    /* accumulate forces for each point */
    if (forces_mat.size() > 0) {
#pragma omp parallel for num_threads(inner_threads) schedule(static)
      for (int j = 0; j < forces_mat[0].size(); j++) {
        vec_type3 acum(0.f, 0.f, 0.f);
        for (int i = 0; i < forces_mat.size(); i++) {
          acum += forces_mat[i][j];
        }
        forces[j] = acum;
      }
    }
/* accumulate force results for this group */
#pragma omp parallel for num_threads(inner_threads)
    for (int i = 0; i < this->total_nucleii(); i++) {
      uint global_atom = this->local2global_nuc[i];
      vec_type3 this_force = forces[i];
      fort_forces(global_atom, 0) += this_force.x;
      fort_forces(global_atom, 1) += this_force.y;
      fort_forces(global_atom, 2) += this_force.z;
    }
  }
  timers.forces.pause();

  timers.rmm.start();
  /* accumulate RMM results for this group */
  if (compute_rmm) {
    const int indexes = this->rmm_bigs.size();
#pragma omp parallel for num_threads(inner_threads) schedule(static)
    for (int i = 0; i < indexes; i++) {
      int bi = this->rmm_bigs[i], row = this->rmm_rows[i],
          col = this->rmm_cols[i];

      double res = (double)cpu_update_rmm(
          function_values_transposed.row(row),
          function_values_transposed.row(col),
          factors_rmm.asArray(), npoints);
      if (my_cdft_vars.do_chrg) {
        const scalar_type* fvr = function_values_transposed.row(row);
        const scalar_type* fvc = function_values_transposed.row(col);
        for (int point = 0; point < npoints; point++) {
          for (int j = 0; j < my_cdft_vars.regions; j++) {
            res += fvr[point] * fvc[point] * factors_cdft(point,j) * my_cdft_vars.Vc(j);
          }
        }
      }
      rmm_global_output(bi) += res;
    }
  }
  timers.rmm.pause();

  energy += localenergy;

#if CPU_RECOMPUTE or !GPU_KERNELS
  /* clear functions */
  gX.deallocate();
  gY.deallocate();
  gZ.deallocate();
  hIX.deallocate();
  hIY.deallocate();
  hIZ.deallocate();
  hPX.deallocate();
  hPY.deallocate();
  hPZ.deallocate();
  function_values_transposed.deallocate();
#endif
}

template <class scalar_type>
void PointGroupCPU<scalar_type>::solve_opened(
    Timers& timers, bool compute_rmm, bool lda, bool compute_forces,
    bool compute_energy, double& energy, double& energy_i, double& energy_c,
    double& energy_c1, double& energy_c2, HostMatrix<double>& fort_forces,
    HostMatrix<double>& rmm_output_local_a,
    HostMatrix<double>& rmm_output_local_b, HostMatrix<double>& becke_dens,
    HostMatrix<double>& becke_spin, CDFTVars& my_cdft_vars) {
  //   std::exit(0);
  int inner_threads = 1;
  const uint group_m = this->total_functions();
  const int npoints = this->points.size();

#if CPU_RECOMPUTE or !GPU_KERNELS
  /** Compute functions **/
  timers.functions.start();
  compute_functions(compute_forces, !lda);
  timers.functions.pause();
#endif

  double localenergy = 0.0;

  // prepare rmm_input for this group
  timers.density.start();

  HostMatrix<scalar_type> rmm_input_a(group_m, group_m), rmm_input_b(group_m, group_m);
  get_rmm_input(rmm_input_a, rmm_input_b);

  vector<vec_type3> forces_a, forces_b;
  vector<std::vector<vec_type3> > forces_mat_a, forces_mat_b;
  HostMatrix<scalar_type> factors_rmm_a, factors_rmm_b, factors_cdft;

  if (compute_rmm || compute_forces) {
    factors_rmm_a.resize(this->points.size(), 1);
    factors_rmm_b.resize(this->points.size(), 1);
    if (my_cdft_vars.do_chrg || my_cdft_vars.do_spin) {
      factors_cdft.resize(this->points.size(), my_cdft_vars.regions);
      factors_cdft.zero();
    }
  }

  if (compute_forces) {
    forces_a.resize(this->total_nucleii(), vec_type3(0.f, 0.f, 0.f));
    forces_b.resize(this->total_nucleii(), vec_type3(0.f, 0.f, 0.f));
    forces_mat_a.resize(
        this->points.size(),
        vector<vec_type3>(this->total_nucleii(), vec_type3(0.f, 0.f, 0.f)));
    forces_mat_b.resize(
        this->points.size(),
        vector<vec_type3>(this->total_nucleii(), vec_type3(0.f, 0.f, 0.f)));
  }

  const int iexch = fortran_vars.iexch;

  /** density **/
  if (lda) {
  } else {
#pragma omp parallel for num_threads(inner_threads) \
    reduction(+ : localenergy) schedule(static)
    for (int point = 0; point < npoints; point++) {
      const scalar_type* fv   = function_values.row(point);
      const scalar_type* gxv  = gX.row(point);
      const scalar_type* gyv  = gY.row(point);
      const scalar_type* gzv  = gZ.row(point);
      const scalar_type* hpxv = hPX.row(point);
      const scalar_type* hpyv = hPY.row(point);
      const scalar_type* hpzv = hPZ.row(point);
      const scalar_type* hixv = hIX.row(point);
      const scalar_type* hiyv = hIY.row(point);
      const scalar_type* hizv = hIZ.row(point);

      GGADensity<scalar_type> da = cpu_compute_density_gga(
          fv, gxv, gyv, gzv, hpxv, hpyv, hpzv, hixv, hiyv, hizv,
          rmm_input_a.asArray(), group_m, rmm_input_a.stride);
      GGADensity<scalar_type> db = cpu_compute_density_gga(
          fv, gxv, gyv, gzv, hpxv, hpyv, hpzv, hixv, hiyv, hizv,
          rmm_input_b.asArray(), group_m, rmm_input_b.stride);

      /** energy / potential **/
      scalar_type exc_corr = 0.0, corr1 = 0.0, corr2 = 0.0;
      scalar_type exc = 0.0, corr = 0.0, y2a = 0.0, y2b = 0.0;
      const vec_type3 dxyz_a(da.tdx, da.tdy, da.tdz),
          dxyz_b(db.tdx, db.tdy, db.tdz);
      const vec_type3 dd1_a(da.tdd1x, da.tdd1y, da.tdd1z),
          dd1_b(db.tdd1x, db.tdd1y, db.tdd1z);
      const vec_type3 dd2_a(da.tdd2x, da.tdd2y, da.tdd2z),
          dd2_b(db.tdd2x, db.tdd2y, db.tdd2z);
      scalar_type pd_a = da.pd, pd_b = db.pd;

      calc_ggaOS<scalar_type, 3>(pd_a, pd_b, dxyz_a, dxyz_b, dd1_a, dd1_b,
                                 dd2_a, dd2_b, exc_corr, exc, corr, corr1,
                                 corr2, y2a, y2b, 9, fortran_vars.fexc);

      const scalar_type wp = this->points[point].weight;

      if (compute_energy) {
        localenergy += ((pd_a + pd_b) * wp) * (exc + corr);

        // Also calculates Becke partition if needed.
        if (fortran_vars.becke) {
          for (int i = 0; i < fortran_vars.atoms; i++) {
            becke_dens(i) += wp * (pd_a + pd_b)
                                * (this->points[point].atom_weights(i));
            becke_spin(i) += wp * (pd_b - pd_a)
                                * (this->points[point].atom_weights(i));
          }
        }
      }

      /** RMM **/
      if (compute_rmm || compute_forces) {
        factors_rmm_a(point) = wp * y2a;
        factors_rmm_b(point) = wp * y2b;

        if (my_cdft_vars.do_chrg || my_cdft_vars.do_spin) {
          for (int i = 0; i < my_cdft_vars.regions; i++) {
            for (int j = 0; j < my_cdft_vars.natom(i); j++) {
              factors_cdft(point,i) = wp
                                  * (this->points[point].atom_weights(my_cdft_vars.atoms(j,i)));
            }
          }
        }        
      }
    }
  }
  timers.density.pause();

  timers.forces.start();
  if (compute_forces) {
    HostMatrix<scalar_type> ddx_a, ddy_a, ddz_a;
    HostMatrix<scalar_type> ddx_b, ddy_b, ddz_b;
    // Build flat per-function atom index once from the shell structure.
    vector<unsigned> func2nuc_vec(group_m);
    for (int i = 0, ii = 0; i < (int)this->total_functions_simple(); i++) {
      unsigned nuc = this->func2local_nuc(ii);
      uint inc = this->small_function_type(i);
      for (uint k = 0; k < inc; k++, ii++) func2nuc_vec[ii] = nuc;
    }

    ddx_a.resize(this->total_nucleii(), 1);
    ddx_b.resize(this->total_nucleii(), 1);
    ddy_a.resize(this->total_nucleii(), 1);
    ddy_b.resize(this->total_nucleii(), 1);
    ddz_a.resize(this->total_nucleii(), 1);
    ddz_b.resize(this->total_nucleii(), 1);

#pragma omp parallel for num_threads(inner_threads)
    for (int point = 0; point < (int)this->points.size(); point++) {
      ddx_a.zero(); ddy_a.zero(); ddz_a.zero();
      ddx_b.zero(); ddy_b.zero(); ddz_b.zero();

      cpu_compute_density_derivs(
          function_values.row(point),
          gX.row(point), gY.row(point), gZ.row(point),
          rmm_input_a.asArray(), group_m,
          func2nuc_vec.data(), this->total_nucleii(),
          ddx_a.data, ddy_a.data, ddz_a.data,
          rmm_input_a.stride);
      cpu_compute_density_derivs(
          function_values.row(point),
          gX.row(point), gY.row(point), gZ.row(point),
          rmm_input_b.asArray(), group_m,
          func2nuc_vec.data(), this->total_nucleii(),
          ddx_b.data, ddy_b.data, ddz_b.data,
          rmm_input_b.stride);

      scalar_type factor_a = factors_rmm_a(point);
      scalar_type factor_b = factors_rmm_b(point);
      for (int i = 0; i < (int)this->total_nucleii(); i++) {
        forces_mat_a[point][i] =
            vec_type3(ddx_a(i), ddy_a(i), ddz_a(i)) * factor_a;
        forces_mat_b[point][i] =
            vec_type3(ddx_b(i), ddy_b(i), ddz_b(i)) * factor_b;
      }
    }

    /* accumulate forces for each point */
    if ((forces_mat_a.size() > 0) && (forces_mat_b.size() > 0)) {
#pragma omp parallel for num_threads(inner_threads) schedule(static)
      for (int j = 0; j < forces_mat_a[0].size(); j++) {
        vec_type3 acum_a(0.f, 0.f, 0.f);
        vec_type3 acum_b(0.f, 0.f, 0.f);
        for (int i = 0; i < forces_mat_a.size(); i++) {
          acum_a += forces_mat_a[i][j];
          acum_b += forces_mat_b[i][j];
        }
        forces_a[j] = acum_a;
        forces_b[j] = acum_b;
      }
    }

/* accumulate force results for this group */
#pragma omp parallel for num_threads(inner_threads)
    for (int i = 0; i < this->total_nucleii(); i++) {
      uint global_atom = this->local2global_nuc[i];
      vec_type3 this_force = forces_a[i] + forces_b[i];
      fort_forces(global_atom, 0) += this_force.x;
      fort_forces(global_atom, 1) += this_force.y;
      fort_forces(global_atom, 2) += this_force.z;
    }
  }
  timers.forces.pause();

  timers.rmm.start();
  /* accumulate RMM results for this group */
  if (compute_rmm) {
    const int indexes = this->rmm_bigs.size();
#pragma omp parallel for num_threads(inner_threads) schedule(static)
    for (int i = 0; i < indexes; i++) {
      int bi = this->rmm_bigs[i], row = this->rmm_rows[i],
          col = this->rmm_cols[i];

      double res_a = (double)cpu_update_rmm(
          function_values_transposed.row(row),
          function_values_transposed.row(col),
          factors_rmm_a.asArray(), npoints);
      double res_b = (double)cpu_update_rmm(
          function_values_transposed.row(row),
          function_values_transposed.row(col),
          factors_rmm_b.asArray(), npoints);

      if (my_cdft_vars.do_chrg || my_cdft_vars.do_spin) {
        const scalar_type* fvr = function_values_transposed.row(row);
        const scalar_type* fvc = function_values_transposed.row(col);
        if (my_cdft_vars.do_chrg) {
          for (int j = 0; j < my_cdft_vars.regions; j++) {
            for (int point = 0; point < npoints; point++) {
              res_a += fvr[point] * fvc[point] * factors_cdft(point,j) * my_cdft_vars.Vc(j);
              res_b += fvr[point] * fvc[point] * factors_cdft(point,j) * my_cdft_vars.Vc(j);
            }
          }
        }
        if (my_cdft_vars.do_spin) {
          for (int j = 0; j < my_cdft_vars.regions; j++) {
            for (int point = 0; point < npoints; point++) {
              res_a -= fvr[point] * fvc[point] * factors_cdft(point,j) * my_cdft_vars.Vs(j);
              res_b += fvr[point] * fvc[point] * factors_cdft(point,j) * my_cdft_vars.Vs(j);
            }
          }
        }
      }

      rmm_output_local_a(bi) += res_a;
      rmm_output_local_b(bi) += res_b;
    }
  }
  timers.rmm.pause();

  energy += localenergy;

#if CPU_RECOMPUTE or !GPU_KERNELS
  /* clear functions */
  gX.deallocate();
  gY.deallocate();
  gZ.deallocate();
  hIX.deallocate();
  hIY.deallocate();
  hIZ.deallocate();
  hPX.deallocate();
  hPY.deallocate();
  hPZ.deallocate();
  function_values_transposed.deallocate();
#endif
}

#if FULL_DOUBLE
template class PointGroup<double>;
template class PointGroupCPU<double>;
#else
template class PointGroup<float>;
template class PointGroupCPU<float>;
#endif
}
