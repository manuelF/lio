#ifndef PARTITION_AUTOTUNE_H
#define PARTITION_AUTOTUNE_H

#include <vector>

// Forward declare double3 — actual definition comes from CUDA or cpu_primitives.
struct double3;

namespace G2G {

struct Point;

// Result of the auto-tuning sweep.
struct AutotuneResult {
  double cube_size;
  double sphere_radius;
  int sphere_decomp;
};

// Evaluate the predicted parallel makespan for a given combination of
// (cube_size, sphere_radius, sphere_decomp).  Creates temporary PointGroups,
// assigns basis functions, and uses compute_optimal_split_cost() to simulate
// the CPU/GPU split.
//
// out_n_groups receives the total number of non-trivial groups created.
double evaluate_partition(double cube_size, double sr, int sphere_decomp,
                          const double3& x0, const double3& x1,
                          const std::vector<Point>& all_points,
                          const std::vector<double>& min_exps_func,
                          const std::vector<double>& min_coeff_func,
                          int n_cpu, int n_gpu, int& out_n_groups);

// Sweep candidate (cube_size, sphere_radius, sphere_decomp) combinations and
// return the triple that minimizes predicted parallel makespan.
//
// Parameters with value < 0 are auto-tuned; values >= 0 are held fixed.
// (sphere_radius == 0 means "no spheres" and is valid as a fixed value.)
AutotuneResult autotune_partition(const double3& x0, const double3& x1,
                                  const std::vector<Point>& all_points,
                                  const std::vector<double>& min_exps_func,
                                  const std::vector<double>& min_coeff_func,
                                  int n_cpu, int n_gpu,
                                  double fixed_cube_size,
                                  double fixed_sphere_radius,
                                  int fixed_sphere_decomp);

}  // namespace G2G

#endif  // PARTITION_AUTOTUNE_H
