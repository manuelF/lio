#include "partition_autotune.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "common.h"
#include "init.h"
#include "partition.h"
#include "partition_cost.h"

using std::vector;

namespace G2G {

double evaluate_partition(double cube_size, double sr, int sphere_decomp,
                          const double3& x0, const double3& x1,
                          const std::vector<Point>& all_points,
                          const std::vector<double>& min_exps_func,
                          const std::vector<double>& min_coeff_func,
                          int n_cpu, int n_gpu, int& out_n_groups) {
  // Separate points into per-atom sphere groups and cube points.
  typedef vector<Point> Group;
  vector<Group> sphere_pts;
  std::vector<Point> cube_pts;
  vector<double> sr_array;

  if (sr > 0) {
    sphere_pts.resize(fortran_vars.atoms);
    sr_array.resize(fortran_vars.atoms);
    for (uint atom = 0; atom < fortran_vars.atoms; atom++) {
      uint atom_shells = fortran_vars.shells(atom);
      uint included_shells = (uint)ceil(sr * atom_shells);
      if (included_shells == 0) {
        sr_array[atom] = 0;
      } else {
        double x = cos((M_PI / (atom_shells + 1)) *
                       (atom_shells - included_shells + 1));
        double rm = fortran_vars.rm(atom);
        sr_array[atom] = rm * (1.0 + x) / (1.0 - x);
      }
    }
  }

  for (size_t p = 0; p < all_points.size(); p++) {
    const Point& pt = all_points[p];
    if (sr > 0) {
      uint atom_shells = fortran_vars.shells(pt.atom);
      uint included_shells = (uint)ceil(sr * atom_shells);
      if (pt.shell >= (atom_shells - included_shells)) {
        sphere_pts[pt.atom].push_back(pt);
        continue;
      }
    }
    cube_pts.push_back(pt);
  }

  // Build cube groups.
  uint3 prism_size = ceil_uint3((x1 - x0) / cube_size);

  vector<vector<vector<Group> > > prism(
      prism_size.x,
      vector<vector<Group> >(prism_size.y, vector<Group>(prism_size.z)));

  for (size_t p = 0; p < cube_pts.size(); p++) {
    uint3 cc = floor_uint3((cube_pts[p].position - x0) / cube_size);
    if (cc.x >= prism_size.x || cc.y >= prism_size.y || cc.z >= prism_size.z)
      continue;
    prism[cc.x][cc.y][cc.z].push_back(cube_pts[p]);
  }

  // Temporarily set the global little_cube_size for assign_functions_as_cube().
  double saved_lcs = little_cube_size;
  little_cube_size = cube_size;

  std::vector<long long> all_pm2;
  int n_groups = 0;

  for (uint i = 0; i < prism_size.x; i++) {
    for (uint j = 0; j < prism_size.y; j++) {
      for (uint k = 0; k < prism_size.z; k++) {
        if (prism[i][j][k].empty()) continue;

        double3 cube_coord_abs = x0 + make_uint3(i, j, k) * cube_size;

        PointGroupCPU<base_scalar_type> cube_tmp;
        for (uint pt = 0; pt < prism[i][j][k].size(); pt++)
          cube_tmp.add_point(prism[i][j][k][pt]);

        cube_tmp.assign_functions_as_cube(cube_coord_abs, min_exps_func,
                                          min_coeff_func);

        if (cube_tmp.total_functions_simple() == 0 ||
            cube_tmp.number_of_points < min_points_per_cube)
          continue;

        all_pm2.push_back((long long)cube_tmp.number_of_points *
                          cube_tmp.total_functions() *
                          cube_tmp.total_functions());
        n_groups++;
      }
    }
  }

  little_cube_size = saved_lcs;

  // Build sphere groups, optionally decomposed into sub-chunks.
  if (sr > 0) {
    for (uint i = 0; i < fortran_vars.atoms; i++) {
      if (sphere_pts[i].empty()) continue;

      const double3& atom_pos = fortran_vars.atom_positions(i);
      Group& all_atom_points = sphere_pts[i];
      size_t n_pts = all_atom_points.size();

      if (sphere_decomp <= 0 || (size_t)sphere_decomp >= n_pts) {
        // No decomposition: one group per atom (original behavior).
        PointGroupCPU<base_scalar_type> sphere_tmp;
        for (size_t p = 0; p < n_pts; p++)
          sphere_tmp.add_point(all_atom_points[p]);
        sphere_tmp.assign_functions_as_sphere(i, sr_array[i], min_exps_func,
                                              min_coeff_func);
        if (sphere_tmp.total_functions_simple() != 0 &&
            sphere_tmp.number_of_points >= min_points_per_cube) {
          all_pm2.push_back((long long)sphere_tmp.number_of_points *
                            sphere_tmp.total_functions() *
                            sphere_tmp.total_functions());
          n_groups++;
        }
      } else {
        // Decompose into sub-chunks with tighter radial bounds.
        for (size_t start = 0; start < n_pts;) {
          size_t end = std::min(start + (size_t)sphere_decomp, n_pts);
          if (n_pts - end < (size_t)sphere_decomp / 4) end = n_pts;

          PointGroupCPU<base_scalar_type> sphere_tmp;
          double max_dist = 0;
          for (size_t p = start; p < end; p++) {
            sphere_tmp.add_point(all_atom_points[p]);
            double d = distance(all_atom_points[p].position, atom_pos);
            if (d > max_dist) max_dist = d;
          }

          sphere_tmp.assign_functions_as_sphere(i, max_dist, min_exps_func,
                                                min_coeff_func);

          if (sphere_tmp.total_functions_simple() != 0 &&
              sphere_tmp.number_of_points >= min_points_per_cube) {
            all_pm2.push_back((long long)sphere_tmp.number_of_points *
                              sphere_tmp.total_functions() *
                              sphere_tmp.total_functions());
            n_groups++;
          }
          start = end;
        }
      }
    }
  }

  out_n_groups = n_groups;
  double makespan = 0.0;
  double sr_ratio = estimate_speed_ratio(gpu_hw);
  compute_optimal_split_cost(all_pm2, n_cpu, n_gpu, sr_ratio, &makespan);
  return makespan;
}

AutotuneResult autotune_partition(const double3& x0, const double3& x1,
                                  const std::vector<Point>& all_points,
                                  const std::vector<double>& min_exps_func,
                                  const std::vector<double>& min_coeff_func,
                                  int n_cpu, int n_gpu,
                                  double fixed_cube_size,
                                  double fixed_sphere_radius,
                                  int fixed_sphere_decomp) {
  const double cs_candidates[] = {4.0, 5.6, 8.0, 11.3, 16.0};
  const double sr_candidates[] = {0.0, 0.3, 0.6, 0.9};
  const int sd_candidates[] = {0, 128, 256, 512};
  const int n_cs = 5, n_sr = 4, n_sd = 4;

  double best_makespan = DBL_MAX;
  AutotuneResult best = {8.0, 0.6, 0};

  for (int ci = 0; ci < n_cs; ci++) {
    double cs =
        (fixed_cube_size > 0) ? fixed_cube_size : cs_candidates[ci];
    for (int si = 0; si < n_sr; si++) {
      double sr =
          (fixed_sphere_radius >= 0) ? fixed_sphere_radius : sr_candidates[si];
      for (int di = 0; di < n_sd; di++) {
        // Skip decomp search if sphere_radius is 0 (no spheres to decompose)
        // or if decomp_size is explicitly set.
        int sd = (sr == 0.0)                ? 0
                 : (fixed_sphere_decomp >= 0) ? fixed_sphere_decomp
                                              : sd_candidates[di];
        int ng = 0;
        double ms =
            evaluate_partition(cs, sr, sd, x0, x1, all_points, min_exps_func,
                               min_coeff_func, n_cpu, n_gpu, ng);
        if (verbose > 3)
          printf(
              "  [partition] cube_size=%.1f sphere_radius=%.1f "
              "sphere_decomp=%d: %d groups, makespan=%.0f\n",
              cs, sr, sd, ng, ms);
        if (ms < best_makespan) {
          best_makespan = ms;
          best.cube_size = cs;
          best.sphere_radius = sr;
          best.sphere_decomp = sd;
        }
        if (sr == 0.0 || fixed_sphere_decomp >= 0) break;
      }
      if (fixed_sphere_radius >= 0) break;
    }
    if (fixed_cube_size > 0) break;
  }

  return best;
}

}  // namespace G2G
