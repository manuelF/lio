#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "../common.h"
#include "../init.h"
#include "../matrix.h"
#include "../partition.h"
#include "cpu_kernels.h"
using namespace std;

namespace G2G {
template <class scalar_type>
void PointGroupCPU<scalar_type>::compute_weights(void) {
  // Pre-extract atom data into flat arrays for cpu_becke_weight_for_point().
  // This is done once per compute_weights() call so the per-point loop stays
  // free of FortranVars accesses and can be called from unit tests.
  const uint total_atoms = fortran_vars.atoms;
  vector<double> ax(total_atoms), ay(total_atoms), az(total_atoms);
  vector<double> arm(total_atoms);
  vector<double> adists(total_atoms * total_atoms);

  for (uint a = 0; a < total_atoms; ++a) {
    const double3& pos = fortran_vars.atom_positions(a);
    ax[a] = pos.x;
    ay[a] = pos.y;
    az[a] = pos.z;
    arm[a] = fortran_vars.rm(a);
  }
  for (uint j = 0; j < total_atoms; ++j)
    for (uint k = 0; k < total_atoms; ++k)
      adists[j * total_atoms + k] = fortran_vars.atom_atom_dists(j, k);

#pragma omp parallel for
  for (int point = 0; point < (int)this->points.size(); point++) {
    uint atom = this->points[point].atom;
    const double3& pp = this->points[point].position;
    bool home_in_local = this->has_nucleii(atom);

    double atom_weight = cpu_becke_weight_for_point(
        pp.x, pp.y, pp.z,
        atom,
        this->local2global_nuc.data(),
        static_cast<uint>(this->total_nucleii()),
        ax.data(), ay.data(), az.data(), arm.data(),
        adists.data(), total_atoms,
        home_in_local);

    this->points[point].weight *= atom_weight;
  }

  if (remove_zero_weights) {
    vector<Point> filteredPoints;
    for (int point = 0; point < (int)this->points.size(); point++) {
      if (this->points[point].weight != 0.0)
        filteredPoints.push_back(this->points[point]);
    }
    this->points.swap(filteredPoints);
    this->number_of_points = this->points.size();
  }
}
#if FULL_DOUBLE
template class PointGroup<double>;
template class PointGroupCPU<double>;
#else
template class PointGroup<float>;
template class PointGroupCPU<float>;
#endif
}
