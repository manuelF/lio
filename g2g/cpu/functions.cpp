#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "../common.h"
#include "../cuda_includes.h"
#include "../init.h"
#include "../matrix.h"
#include "../partition.h"
#include "cpu_kernels.h"
using namespace std;

namespace G2G {

template <class scalar_type>
void PointGroupCPU<scalar_type>::compute_functions(bool forces, bool gga) {
#if !CPU_RECOMPUTE && GPU_KERNELS
  if (this->inGlobal) return;
  this->inGlobal = true;
  forces = gga = true;  // Vamos a cachear asi que guardemos todo y listo
#endif
  /* Load group functions */
  uint group_m = this->total_functions();
  int numpoints = this->number_of_points;

  function_values.resize(group_m, numpoints);
  if (forces || gga) {
    gX.resize(group_m, numpoints);
    gX.zero();
    gY.resize(group_m, numpoints);
    gY.zero();
    gZ.resize(group_m, numpoints);
    gZ.zero();
  }
  if (gga) {
    hPX.resize(group_m, numpoints);
    hPX.zero();
    hPY.resize(group_m, numpoints);
    hPY.zero();
    hPZ.resize(group_m, numpoints);
    hPZ.zero();
    hIX.resize(group_m, numpoints);
    hIX.zero();
    hIY.resize(group_m, numpoints);
    hIY.zero();
    hIZ.resize(group_m, numpoints);
    hIZ.zero();
  }

#pragma omp parallel for schedule(static)
  for (int point = 0; point < (int)this->points.size(); point++) {
    vec_type3 point_position = vec_type3(this->points[point].position.x,
                                         this->points[point].position.y,
                                         this->points[point].position.z);

    for (uint i = 0, ii = 0; i < this->total_functions_simple(); i++) {
      // Determine shell type: 0=S, 1=P, 2=D
      int shell_type;
      if      (i < this->s_functions)                          shell_type = 0;
      else if (i < this->s_functions + this->p_functions)      shell_type = 1;
      else                                                      shell_type = 2;

      // Displacement and squared distance from atom to integration point
      uint nuc = this->func2global_nuc(i);
      const vec_type3 v(point_position -
                        vec_type3(fortran_vars.atom_positions(nuc)));
      scalar_type dist2 = v.length2();

      // Gather contraction data
      uint global_func = this->local2global_func[i];
      uint nc = fortran_vars.contractions(global_func);
      // alphas and coeffs as stack arrays (MAX_CONTRACTIONS is small, ~13)
      scalar_type alphas[MAX_CONTRACTIONS], coeffs[MAX_CONTRACTIONS];
      for (uint c = 0; c < nc; ++c) {
        alphas[c] = fortran_vars.a_values(global_func, c);
        coeffs[c] = fortran_vars.c_values(global_func, c);
      }

      scalar_type norm = fortran_vars.normalization_factor;

      // Output arrays on the stack (max 6 functions per simple shell)
      scalar_type val[6], gx[6], gy[6], gz[6];
      scalar_type hpx[6], hpy[6], hpz[6], hix[6], hiy[6], hiz[6];
      bool compute_grad = forces || gga;
      bool compute_hess = gga;

      int nf = cpu_eval_gto_shell<scalar_type>(
          v.x, v.y, v.z, dist2,
          alphas, coeffs, (int)nc,
          shell_type, norm,
          compute_grad, compute_hess,
          val,
          compute_grad ? gx : nullptr, compute_grad ? gy : nullptr, compute_grad ? gz : nullptr,
          compute_hess ? hpx : nullptr, compute_hess ? hpy : nullptr, compute_hess ? hpz : nullptr,
          compute_hess ? hix : nullptr, compute_hess ? hiy : nullptr, compute_hess ? hiz : nullptr);

      // Write outputs into the group matrices
      for (int k = 0; k < nf; ++k) {
        function_values(ii + k, point) = val[k];
        if (forces || gga) {
          gX(ii + k, point) = gx[k];
          gY(ii + k, point) = gy[k];
          gZ(ii + k, point) = gz[k];
        }
        if (gga) {
          hPX(ii + k, point) = hpx[k];
          hPY(ii + k, point) = hpy[k];
          hPZ(ii + k, point) = hpz[k];
          hIX(ii + k, point) = hix[k];
          hIY(ii + k, point) = hiy[k];
          hIZ(ii + k, point) = hiz[k];
        }
      }
      ii += nf;
    }
  }

  function_values.transpose(function_values_transposed);
}

#if FULL_DOUBLE
template class PointGroupCPU<double>;
#else
template class PointGroupCPU<float>;
#endif
}
