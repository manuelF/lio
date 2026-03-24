/* includes */
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <cfloat>
#include <cassert>

#include "common.h"
#include "init.h"
#include "partition.h"

using namespace std;
using namespace G2G;

/************************************************************
 * Evaluate a (cube_size, sphere_radius) pair: separate all
 * points into sphere vs cube groups, assign basis functions,
 * and return the predicted parallel makespan.
 ************************************************************/
static double evaluate_partition(
    double cube_size, double sr, int sphere_decomp,
    const double3& x0, const double3& x1,
    const std::vector<Point>& all_points,
    const std::vector<double>& min_exps_func,
    const std::vector<double>& min_coeff_func,
    int n_cpu, int n_gpu,
    int& out_n_groups) {

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

        all_pm2.push_back(
            (long long)cube_tmp.number_of_points *
            cube_tmp.total_functions() * cube_tmp.total_functions());
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
        sphere_tmp.assign_functions_as_sphere(i, sr_array[i],
                                              min_exps_func, min_coeff_func);
        if (sphere_tmp.total_functions_simple() != 0 &&
            sphere_tmp.number_of_points >= min_points_per_cube) {
          all_pm2.push_back(
              (long long)sphere_tmp.number_of_points *
              sphere_tmp.total_functions() * sphere_tmp.total_functions());
          n_groups++;
        }
      } else {
        // Decompose into sub-chunks with tighter radial bounds.
        for (size_t start = 0; start < n_pts; ) {
          size_t end = std::min(start + (size_t)sphere_decomp, n_pts);
          if (n_pts - end < (size_t)sphere_decomp / 4) end = n_pts;

          PointGroupCPU<base_scalar_type> sphere_tmp;
          double max_dist = 0;
          for (size_t p = start; p < end; p++) {
            sphere_tmp.add_point(all_atom_points[p]);
            double d = distance(all_atom_points[p].position, atom_pos);
            if (d > max_dist) max_dist = d;
          }

          sphere_tmp.assign_functions_as_sphere(i, max_dist,
                                                min_exps_func, min_coeff_func);

          if (sphere_tmp.total_functions_simple() != 0 &&
              sphere_tmp.number_of_points >= min_points_per_cube) {
            all_pm2.push_back(
                (long long)sphere_tmp.number_of_points *
                sphere_tmp.total_functions() * sphere_tmp.total_functions());
            n_groups++;
          }
          start = end;
        }
      }
    }
  }

  out_n_groups = n_groups;
  double makespan = 0.0;
  compute_optimal_split_cost(all_pm2, n_cpu, n_gpu, &makespan);
  return makespan;
}

/************************************************************
 * Construct partition
 ************************************************************/

// Sorting the cubes in increasing order of size in bytes in GPU.
template <typename T>
bool comparison_by_size(const T& a, const T& b) {
  return a.size_in_gpu() < b.size_in_gpu();
}

template <typename T>
void sortBySize(std::vector<T>& input) {
  sort(input.begin(), input.end(), comparison_by_size<T>);
}

void load_pools(const vector<int>& elements, const vector<vector<int> >& work,
                vector<int>& pool_sizes) {
  pool_sizes.clear();
  for (uint i = 0; i < work.size(); i++) {
    int largest_pool = 0;
    for (uint j = 0; j < work[i].size(); j++) {
      largest_pool = max(largest_pool, elements[work[i][j]]);
    }
    pool_sizes.push_back(largest_pool);
  }
}

template <typename T>
long long total_costs(const vector<T*>& elements) {
  long long res = 0;
  for (uint i = 0; i < elements.size(); i++) res += elements[i]->cost();
  return res;
}

int split_bins(const vector<pair<long long, int> >& costs,
               vector<vector<int> >& workloads, long long capacity) {
  // Bin Packing heuristic
  workloads.clear();
  for (uint i = 0; i < costs.size(); i++) {
    int next_bin = -1;
    for (uint j = 0; j < workloads.size(); j++) {
      long long slack = capacity;
      for (uint k = 0; k < workloads[j].size(); k++) {
        slack -= costs[workloads[j][k]].first;
      }
      if (slack >= costs[i].first && next_bin == -1) {
        next_bin = j;
        break;
      }
    }
    if (next_bin == -1) {
      if (capacity < costs[i].second) {
        return INT_MAX;
      }
      next_bin = workloads.size();
      workloads.push_back(vector<int>());
    }
    workloads[next_bin].push_back(i);
  }
  return workloads.size();
}

void Partition::compute_work_partition() {
  if (G2G::cpu_threads == 0) return;
  vector<pair<long long, int> > costs;
  for (uint i = 0; i < cubes.size(); i++)
    if (!cubes[i]->is_big_group())
      costs.push_back(make_pair(cubes[i]->cost(), i));

  const uint ncubes = cubes.size();
  for (uint i = 0; i < spheres.size(); i++)
    if (!spheres[i]->is_big_group())
      costs.push_back(make_pair(spheres[i]->cost(), ncubes + i));

  if (costs.empty()) return;

  sort(costs.begin(), costs.end());
  reverse(costs.begin(), costs.end());

  long long min_cost = costs.front().second - 1,
            max_cost = total_costs(cubes) + total_costs(spheres) + 1;

  while (max_cost - min_cost > 1) {
    long long candidate = min_cost + (max_cost - min_cost) / 2;

    vector<vector<int> > workloads;
    int bins = split_bins(costs, workloads, candidate);
    if (bins <= G2G::cpu_threads) {
      max_cost = candidate;
    } else {
      min_cost = candidate;
    }
  }

  split_bins(costs, work, max_cost);
  for (uint i = 0; i < work.size(); i++) sort(work[i].begin(), work[i].end());

  double maxp = 0, minp = total_costs(cubes) + total_costs(spheres) + 1;
  for (uint i = 0; i < work.size(); i++) {
    long long total = 0;
    for (uint j = 0; j < work[i].size(); j++) {
      long long c = costs[work[i][j]].first;
      work[i][j] = costs[work[i][j]].second;
      total += c;
    }
    if (minp > total) minp = total;
    if (maxp < total) maxp = total;
    if (verbose > 4) printf("  Partition %d: %lld\n", i, total);
  }
  if (verbose > 4) printf("  MAX/MIN ratio: %lf\n", maxp / minp);
}

int getintenv(const char* str, int default_value) {
  char* v = getenv(str);
  if (v == NULL) return default_value;
  int ret = strtol(v, NULL, 10);
  return ret;
}

void diagnostic() {
  printf("  Threads OMP: %d - Threads CPU: %d - Threads GPU: %d\n",
         omp_get_max_threads(), G2G::cpu_threads, G2G::gpu_threads);
  printf("  Small cube correction: %d - Split cost (P*M^2): %lld%s\n",
         MINCOST, (long long)G2G::SPLIT_COST,
         getenv("LIO_SPLIT_COST") ? " (manual)" : " (auto)");
}

template <class T>
bool is_big_group(const T& points) {
  assert(G2G::cpu_threads > 0 || G2G::gpu_threads > 0);
  if (G2G::cpu_threads == 0) return true;
  if (G2G::gpu_threads == 0) return false;
  return (points.size() > G2G::SPLITPOINTS);
}

struct Sorter {
  template <class T>
  bool operator()(const T l, const T r) {
    return (l->cost() < r->cost());
  }
};

/* methods */
void Partition::regenerate(void) {
  Timer tweights;

  // Environment variable overrides.
  char* lcs_env = getenv("LIO_CUBE_SIZE");
  if (lcs_env) little_cube_size = atof(lcs_env);
  char* sr_env = getenv("LIO_SPHERE_RADIUS");
  if (sr_env) sphere_radius = atof(sr_env);
  char* sd_env = getenv("LIO_SPHERE_DECOMP");
  if (sd_env) sphere_decomp_size = atoi(sd_env);

  // Determina el exponente minimo para cada tipo de atomo.
  // uno por elemento de la tabla periodica.
  vector<double> min_exps(120, numeric_limits<double>::max());
  for (uint i = 0; i < fortran_vars.m; i++) {
    uint contractions = fortran_vars.contractions(i);
    uint nuc = fortran_vars.nucleii(i) - 1;
    uint nuc_type = fortran_vars.atom_types(nuc);
    for (uint j = 0; j < contractions; j++) {
      min_exps[nuc_type] = min(min_exps[nuc_type], fortran_vars.a_values(i, j));
    }
  }

  // Un exponente y un coeficiente por funcion.
  vector<double> min_exps_func(fortran_vars.m, numeric_limits<double>::max());
  vector<double> min_coeff_func(fortran_vars.m);
  for (uint i = 0; i < fortran_vars.m; i++) {
    uint contractions = fortran_vars.contractions(i);
    for (uint j = 0; j < contractions; j++) {
      if (fortran_vars.a_values(i, j) < min_exps_func[i]) {
        min_exps_func[i] = fortran_vars.a_values(i, j);
        min_coeff_func[i] = fortran_vars.c_values(i, j);
      }
    }
  }

  // Encontrando el prisma conteniendo el sistema.
  double3 x0 = make_double3(0, 0, 0);
  double3 x1 = make_double3(0, 0, 0);
  for (uint atom = 0; atom < fortran_vars.atoms; atom++) {
    double3 atom_position(fortran_vars.atom_positions(atom));
    uint atom_type = fortran_vars.atom_types(atom);
    // TODO: max_radios esta al doble de lo que deberia porque el criterio no
    // esta bien aplicado.
    double max_radius = 2 * sqrt(max_function_exponent / min_exps[atom_type]);
    _DBG(cout << "tipo: " << atom_type << " " << min_exps[atom_type]
              << " radio: " << max_radius << endl);
    double3 tuple_max_radius = make_double3(max_radius, max_radius, max_radius);
    if (atom == 0) {
      x0 = atom_position - tuple_max_radius;
      x1 = atom_position + tuple_max_radius;
    } else {
      x0.x = min(x0.x, atom_position.x - max_radius);
      x0.y = min(x0.y, atom_position.y - max_radius);
      x0.z = min(x0.z, atom_position.z - max_radius);

      x1.x = max(x1.x, atom_position.x + max_radius);
      x1.y = max(x1.y, atom_position.y + max_radius);
      x1.z = max(x1.z, atom_position.z + max_radius);
    }
  }

  // El prisma tiene vertices (x,y), con x0 el vertice inferior, izquierdo y mas
  // lejano
  // y x1 el vertice superior, derecho y mas cercano.

  typedef vector<Point> Group;

  // Precomputamos las distancias entre atomos.
  for (uint i = 0; i < fortran_vars.atoms; i++) {
    const double3& atom_i_position(fortran_vars.atom_positions(i));
    double nearest_neighbor_dist = numeric_limits<double>::max();

    for (uint j = 0; j < fortran_vars.atoms; j++) {
      const double3& atom_j_position(fortran_vars.atom_positions(j));
      double dist = length(atom_i_position - atom_j_position);
      fortran_vars.atom_atom_dists(i, j) = dist;
      if (i != j) nearest_neighbor_dist = min(nearest_neighbor_dist, dist);
    }
    fortran_vars.nearest_neighbor_dists(i) = nearest_neighbor_dist;
  }

  // =========================================================================
  // Point generation pass: generate ALL points into a flat vector.
  // Sphere/cube separation happens later, after auto-tuning sphere_radius.
  // =========================================================================
  uint puntos_totales = 0;
  uint puntos_finales = 0;
  uint funciones_finales = 0;
  uint costo = 0;

  // Limpiamos las colecciones de las esferas y cubos que tengamos guardadas.
  this->clear();

  std::vector<Point> all_points;

  for (uint atom = 0; atom < fortran_vars.atoms; atom++) {
    uint atom_shells = fortran_vars.shells(atom);
    const double3& atom_position(fortran_vars.atom_positions(atom));

    double t0 = M_PI / (atom_shells + 1);
    double rm = fortran_vars.rm(atom);

    puntos_totales += (uint)fortran_vars.grid_size * atom_shells;
    for (uint shell = 0; shell < atom_shells; shell++) {
      double t1 = t0 * (shell + 1);
      double x = cos(t1);
      double w = t0 * abs(sin(t1));
      double r1 = rm * (1.0 + x) / (1.0 - x);
      double wrad = w * (r1 * r1) * rm * 2.0 / ((1.0 - x) * (1.0 - x));

      for (uint point = 0; point < (uint)fortran_vars.grid_size; point++) {
        double3 rel_point_position =
            make_double3(fortran_vars.e(point, 0), fortran_vars.e(point, 1),
                         fortran_vars.e(point, 2));
        double3 point_position = atom_position + rel_point_position * r1;
        bool inside_prism =
            ((x0.x <= point_position.x && point_position.x <= x1.x) &&
             (x0.y <= point_position.y && point_position.y <= x1.y) &&
             (x0.z <= point_position.z && point_position.z <= x1.z));
        if (inside_prism) {
          double point_weight =
              wrad * fortran_vars.wang(point);  // integration weight
          Point point_object(atom, shell, point, point_position, point_weight);
          all_points.push_back(point_object);
        }
      }
    }
  }

  // =========================================================================
  // Auto-tune little_cube_size, sphere_radius, and sphere_decomp_size:
  // try multiple (cube_size, sphere_radius, decomp_size) triples, pick the
  // one that minimizes predicted parallel makespan.
  // Activated when little_cube_size < 0 or sphere_radius < 0.
  // =========================================================================
  bool do_autotune = (little_cube_size < 0 || sphere_radius < 0);
#if GPU_KERNELS
  if (G2G::cpu_threads > 0 && G2G::gpu_threads > 0 && do_autotune) {
    const double cs_candidates[] = {4.0, 5.6, 8.0, 11.3, 16.0};
    const double sr_candidates[] = {0.0, 0.3, 0.6, 0.9};
    const int sd_candidates[] = {0, 128, 256, 512};
    const int n_cs = 5, n_sr = 4, n_sd = 4;
    double best_makespan = DBL_MAX;
    double best_cs = 8.0, best_sr = 0.6;
    int best_sd = 0;

    for (int ci = 0; ci < n_cs; ci++) {
      double cs = (little_cube_size > 0) ? little_cube_size : cs_candidates[ci];
      for (int si = 0; si < n_sr; si++) {
        double sr = (sphere_radius >= 0) ? sphere_radius : sr_candidates[si];
        for (int di = 0; di < n_sd; di++) {
          // Skip decomp search if sphere_radius is 0 (no spheres to decompose)
          // or if decomp_size is explicitly set.
          int sd = (sr == 0.0) ? 0
                 : (sphere_decomp_size >= 0) ? sphere_decomp_size
                 : sd_candidates[di];
          int ng = 0;
          double ms = evaluate_partition(cs, sr, sd, x0, x1, all_points,
                                         min_exps_func, min_coeff_func,
                                         cpu_threads, gpu_threads, ng);
          if (verbose > 3)
            printf("  [partition] cube_size=%.1f sphere_radius=%.1f "
                   "sphere_decomp=%d: %d groups, makespan=%.0f\n",
                   cs, sr, sd, ng, ms);
          if (ms < best_makespan) {
            best_makespan = ms;
            best_cs = cs;
            best_sr = sr;
            best_sd = sd;
          }
          if (sr == 0.0 || sphere_decomp_size >= 0) break;
        }
        if (sphere_radius >= 0) break;
      }
      if (little_cube_size > 0) break;
    }
    if (little_cube_size < 0) little_cube_size = best_cs;
    if (sphere_radius < 0) sphere_radius = best_sr;
    if (sphere_decomp_size < 0) sphere_decomp_size = best_sd;
  }
#endif
  if (little_cube_size < 0) little_cube_size = 8.0;
  if (sphere_radius < 0) sphere_radius = 0.6;
  if (sphere_decomp_size < 0) sphere_decomp_size = 0;

  if (do_autotune && verbose > 0)
    printf("  Auto-detected cube_size=%.1f sphere_radius=%.1f "
           "sphere_decomp=%d\n",
           little_cube_size, sphere_radius, sphere_decomp_size);

  // =========================================================================
  // Separate points into sphere vs cube using the (possibly auto-tuned) params.
  // =========================================================================
  vector<Group> sphere_points;
  vector<double> sphere_radius_array;
  if (sphere_radius > 0) {
    sphere_radius_array.resize(fortran_vars.atoms);
    sphere_points.resize(fortran_vars.atoms);
    for (uint atom = 0; atom < fortran_vars.atoms; atom++) {
      uint atom_shells = fortran_vars.shells(atom);
      uint included_shells = (uint)ceil(sphere_radius * atom_shells);
      double radius;
      if (included_shells == 0) {
        radius = 0;
      } else {
        double x = cos((M_PI / (atom_shells + 1)) *
                       (atom_shells - included_shells + 1));
        double rm = fortran_vars.rm(atom);
        radius = rm * (1.0 + x) / (1.0 - x);
      }
      sphere_radius_array[atom] = radius;
    }
  }

  uint3 prism_size = ceil_uint3((x1 - x0) / little_cube_size);

  vector<vector<vector<Group> > > prism(
      prism_size.x,
      vector<vector<Group> >(prism_size.y, vector<Group>(prism_size.z)));

  for (size_t p = 0; p < all_points.size(); p++) {
    const Point& pt = all_points[p];
    if (sphere_radius > 0) {
      uint atom_shells = fortran_vars.shells(pt.atom);
      uint included_shells = (uint)ceil(sphere_radius * atom_shells);
      if (pt.shell >= (atom_shells - included_shells)) {
        sphere_points[pt.atom].push_back(pt);
        continue;
      }
    }
    uint3 cube_coord =
        floor_uint3((pt.position - x0) / little_cube_size);
    if (cube_coord.x >= prism_size.x || cube_coord.y >= prism_size.y ||
        cube_coord.z >= prism_size.z)
      throw std::runtime_error("Se accedio a un cubo invalido");
    prism[cube_coord.x][cube_coord.y][cube_coord.z].push_back(pt);
  }

  G2G::MINCOST = getintenv("LIO_MINCOST_OFFSET", 250000);
  G2G::THRESHOLD = getintenv("LIO_SPLIT_THRESHOLD", 80);
  G2G::SPLITPOINTS = getintenv("LIO_SPLIT_POINTS", 200);
  uint nco_m = 0;
  uint m_m = 0;
  puntos_finales = 0;

  // =========================================================================
  // Phase 1: Create all group temporaries as PointGroupCPU.
  // This gives us P and M for every group before deciding CPU/GPU assignment.
  // =========================================================================
  std::vector<PointGroupCPU<base_scalar_type>> cube_temps;
  for (uint i = 0; i < prism_size.x; i++) {
    for (uint j = 0; j < prism_size.y; j++) {
      for (uint k = 0; k < prism_size.z; k++) {
        double3 cube_coord_abs = x0 + make_uint3(i, j, k) * little_cube_size;

        PointGroupCPU<base_scalar_type> cube_tmp;
        for (uint point = 0; point < prism[i][j][k].size(); point++)
          cube_tmp.add_point(prism[i][j][k][point]);

        cube_tmp.assign_functions_as_cube(cube_coord_abs, min_exps_func,
                                          min_coeff_func);

        if ((cube_tmp.total_functions_simple() == 0) ||
            (cube_tmp.number_of_points < min_points_per_cube)) {
          continue;
        }
        cube_temps.push_back(cube_tmp);
      }
    }
  }

  std::vector<PointGroupCPU<base_scalar_type>> sphere_temps;
  if (sphere_radius > 0) {
    const int sd = sphere_decomp_size;
    long long cost_orig = 0, cost_decomp = 0;
    int atoms_with_spheres = 0, total_subgroups = 0;

    for (uint i = 0; i < fortran_vars.atoms; i++) {
      Group& sphere_i = sphere_points[i];
      if (sphere_i.empty()) continue;

      const double3& atom_pos = fortran_vars.atom_positions(i);
      size_t n_pts = sphere_i.size();
      atoms_with_spheres++;

      if (sd <= 0 || (size_t)sd >= n_pts) {
        // No decomposition: one group per atom (original behavior).
        PointGroupCPU<base_scalar_type> sphere_tmp;
        for (size_t p = 0; p < n_pts; p++)
          sphere_tmp.add_point(sphere_i[p]);
        sphere_tmp.assign_functions_as_sphere(i, sphere_radius_array[i],
                                              min_exps_func, min_coeff_func);
        if (sphere_tmp.total_functions_simple() != 0 &&
            sphere_tmp.number_of_points >= min_points_per_cube) {
          long long c = (long long)sphere_tmp.number_of_points *
                        sphere_tmp.total_functions() * sphere_tmp.total_functions();
          cost_orig += c;
          cost_decomp += c;
          sphere_temps.push_back(sphere_tmp);
          total_subgroups++;
        }
      } else {
        // Compute baseline cost for logging.
        PointGroupCPU<base_scalar_type> baseline_tmp;
        for (size_t p = 0; p < n_pts; p++)
          baseline_tmp.add_point(sphere_i[p]);
        baseline_tmp.assign_functions_as_sphere(i, sphere_radius_array[i],
                                                min_exps_func, min_coeff_func);
        int baseline_M = baseline_tmp.total_functions();
        cost_orig += (long long)n_pts * baseline_M * baseline_M;

        int atom_subgroups = 0;
        for (size_t start = 0; start < n_pts; ) {
          size_t end = std::min(start + (size_t)sd, n_pts);
          if (n_pts - end < (size_t)sd / 4) end = n_pts;

          PointGroupCPU<base_scalar_type> sphere_tmp;
          double max_dist = 0;
          for (size_t p = start; p < end; p++) {
            sphere_tmp.add_point(sphere_i[p]);
            double d = distance(sphere_i[p].position, atom_pos);
            if (d > max_dist) max_dist = d;
          }

          sphere_tmp.assign_functions_as_sphere(i, max_dist,
                                                min_exps_func, min_coeff_func);

          if (sphere_tmp.total_functions_simple() != 0 &&
              sphere_tmp.number_of_points >= min_points_per_cube) {
            int sub_M = sphere_tmp.total_functions();
            long long sub_cost = (long long)sphere_tmp.number_of_points * sub_M * sub_M;
            cost_decomp += sub_cost;
            if (verbose > 3)
              printf("  [sphere-decomp] atom %u chunk %d: %u pts, "
                     "dist=%.2f (vs %.2f), M=%d (vs %d), cost=%lld\n",
                     i, atom_subgroups,
                     sphere_tmp.number_of_points,
                     max_dist, sphere_radius_array[i],
                     sub_M, baseline_M, sub_cost);
            sphere_temps.push_back(sphere_tmp);
            atom_subgroups++;
          }
          start = end;
        }
        total_subgroups += atom_subgroups;
      }
    }
    if (verbose > 0) {
      printf("  [sphere-decomp] %d atoms -> %d subgroups (decomp_size=%d)\n",
             atoms_with_spheres, total_subgroups, sd);
      if (sd > 0)
        printf("  [sphere-decomp] cost: original=%lld decomposed=%lld "
               "reduction=%.1f%%\n",
               cost_orig, cost_decomp,
               cost_orig > 0 ? 100.0 * (1.0 - (double)cost_decomp / cost_orig) : 0.0);
    }
  }

  // =========================================================================
  // Phase 2: Compute optimal CPU/GPU split threshold.
  // Uses LPT simulation to balance max(CPU_bottleneck, GPU_total).
  // LIO_SPLIT_COST env var overrides auto-tuning (for benchmarking).
  // =========================================================================
#if GPU_KERNELS
  {
    char* sc = getenv("LIO_SPLIT_COST");
    if (sc) {
      G2G::SPLIT_COST = strtoll(sc, NULL, 10);
    } else {
      std::vector<long long> all_pm2;
      all_pm2.reserve(cube_temps.size() + sphere_temps.size());
      for (size_t i = 0; i < cube_temps.size(); i++) {
        all_pm2.push_back(
            (long long)cube_temps[i].number_of_points *
            cube_temps[i].total_functions() * cube_temps[i].total_functions());
      }
      for (size_t i = 0; i < sphere_temps.size(); i++) {
        all_pm2.push_back(
            (long long)sphere_temps[i].number_of_points *
            sphere_temps[i].total_functions() *
            sphere_temps[i].total_functions());
      }
      G2G::SPLIT_COST =
          compute_optimal_split_cost(all_pm2, G2G::cpu_threads,
                                     G2G::gpu_threads);
    }
  }
#endif

  // =========================================================================
  // Phase 3: Convert each temp to its final CPU or GPU type.
  // =========================================================================
  for (size_t ti = 0; ti < cube_temps.size(); ti++) {
    PointGroup<base_scalar_type>* cube;
#if GPU_KERNELS
    if (should_use_gpu(cube_temps[ti].number_of_points,
                       cube_temps[ti].total_functions())) {
      cube = new PointGroupGPU<base_scalar_type>();
      cube->move_base_from(cube_temps[ti]);
    } else {
      cube = new PointGroupCPU<base_scalar_type>(cube_temps[ti]);
    }
#else
    cube = new PointGroupCPU<base_scalar_type>(cube_temps[ti]);
#endif

    tweights.start();
    cube->compute_weights();
    tweights.pause();

    if (cube->number_of_points < min_points_per_cube) {
      cout << "CUBE: not enough points" << endl;
      delete cube;
      continue;
    }
    cubes.push_back(cube);

    puntos_finales += cube->number_of_points;
    funciones_finales += cube->number_of_points * cube->total_functions();
    costo += cube->number_of_points *
             (cube->total_functions() * cube->total_functions());
    nco_m += cube->total_functions() * fortran_vars.nco;
    m_m += cube->total_functions() * cube->total_functions();
  }

  if (sphere_radius > 0) {
    for (size_t ti = 0; ti < sphere_temps.size(); ti++) {
      PointGroup<base_scalar_type>* sphere;
#if GPU_KERNELS
      if (should_use_gpu(sphere_temps[ti].number_of_points,
                         sphere_temps[ti].total_functions())) {
        sphere = new PointGroupGPU<base_scalar_type>();
        sphere->move_base_from(sphere_temps[ti]);
      } else {
        sphere = new PointGroupCPU<base_scalar_type>(sphere_temps[ti]);
      }
#else
      sphere = new PointGroupCPU<base_scalar_type>(sphere_temps[ti]);
#endif

      assert(sphere->number_of_points != 0);
      tweights.start();
      sphere->compute_weights();
      tweights.pause();
      if (sphere->number_of_points < min_points_per_cube) {
        cout << "not enough points" << endl;
        delete sphere;
        continue;
      }
      assert(sphere->number_of_points != 0);
      spheres.push_back(sphere);

      puntos_finales += sphere->number_of_points;
      funciones_finales += sphere->number_of_points * sphere->total_functions();
      costo += sphere->number_of_points *
               (sphere->total_functions() * sphere->total_functions());
      nco_m += sphere->total_functions() * fortran_vars.nco;
      m_m += sphere->total_functions() * sphere->total_functions();
    }
  }

  // TODO fix these sorts now that spheres and cubes are pointers
  sort(spheres.begin(), spheres.end(), Sorter());
  sort(cubes.begin(), cubes.end(), Sorter());

  // Initialize the global memory pool for CUDA.
  // When free_global_memory < 0 (the default sentinel), auto-detect the optimal
  // cache budget based on available GPU memory and total cache needs.
  double effective_fgm = G2G::free_global_memory;
#if GPU_KERNELS
  if (effective_fgm < 0.0) {
    // Auto-detect: compute total cache need for all GPU groups
    size_t total_cache_need = 0;
    uint gpu_group_count = 0;
    for (uint i = 0; i < cubes.size(); i++) {
      if (cubes[i]->is_big_group()) {
        total_cache_need += cubes[i]->size_in_gpu();
        gpu_group_count++;
      }
    }
    for (uint i = 0; i < spheres.size(); i++) {
      if (spheres[i]->is_big_group()) {
        total_cache_need += spheres[i]->size_in_gpu();
        gpu_group_count++;
      }
    }

    size_t free_mem = 0, total_mem = 0;
    cudaGetMemoryInfo(free_mem, total_mem);

    // Reserve headroom for per-group temporaries (partial_densities, dxyz, dd1,
    // dd2, factors, rmm_output, textures, etc.) and system overhead.
    // Empirical: ~20% of free memory or at least 200 MB.
    size_t headroom = max((size_t)(free_mem * 0.2), (size_t)(200 * 1024 * 1024));

    if (free_mem > headroom && total_cache_need > 0) {
      size_t cache_budget = free_mem - headroom;
      if (total_cache_need <= cache_budget) {
        // Everything fits — set fgm to exactly cover all groups
        effective_fgm = (double)total_cache_need / (double)free_mem;
        // Add a small margin so rounding doesn't cause the last group to miss
        effective_fgm = min(effective_fgm * 1.05, 0.8);
      } else {
        // Partial caching — use available budget
        effective_fgm = (double)cache_budget / (double)free_mem;
      }
    } else {
      effective_fgm = 0.0;  // No GPU memory available for caching
    }

    if (verbose > 0) {
      printf("  Auto-detected fgm=%.3f (cache need: %.1f MB, free: %.1f MB, "
             "headroom: %.1f MB, %u GPU groups)\n",
             effective_fgm,
             total_cache_need / (1024.0 * 1024.0),
             free_mem / (1024.0 * 1024.0),
             headroom / (1024.0 * 1024.0),
             gpu_group_count);
    }
  }
#else
  if (effective_fgm < 0.0) effective_fgm = 0.0;
#endif
  GlobalMemoryPool::init(effective_fgm);

  if (timer_single) cout << "  Weights: " << tweights << endl;

  for (uint i = 0; i < cubes.size(); i++) {
    cubes[i]->compute_indexes();
  }
  for (uint i = 0; i < spheres.size(); i++) {
    spheres[i]->compute_indexes();
  }

  compute_work_partition();

  // Print group size distribution for CPU/GPU balance analysis
  if (verbose > 3 && G2G::cpu_threads > 0 && G2G::gpu_threads > 0) {
    int cpu_count = 0, gpu_count = 0;
    long long cpu_cost = 0, gpu_cost = 0;
    std::vector<int> gpu_sizes;
    for (uint i = 0; i < cubes.size(); i++) {
      if (cubes[i]->is_big_group()) {
        gpu_count++;
        gpu_cost += cubes[i]->cost();
        gpu_sizes.push_back(cubes[i]->number_of_points);
      } else {
        cpu_count++;
        cpu_cost += cubes[i]->cost();
      }
    }
    for (uint i = 0; i < spheres.size(); i++) {
      if (spheres[i]->is_big_group()) {
        gpu_count++;
        gpu_cost += spheres[i]->cost();
        gpu_sizes.push_back(spheres[i]->number_of_points);
      } else {
        cpu_count++;
        cpu_cost += spheres[i]->cost();
      }
    }
    std::sort(gpu_sizes.begin(), gpu_sizes.end());
    printf("  [partition] SPLIT_COST=%lld (P*M^2)  CPU: %d groups (cost %lld)  "
           "GPU: %d groups (cost %lld)\n",
           (long long)SPLIT_COST, cpu_count, cpu_cost, gpu_count, gpu_cost);
    if (!gpu_sizes.empty()) {
      printf("  [partition] GPU group points: min=%d median=%d max=%d\n",
             gpu_sizes.front(),
             gpu_sizes[gpu_sizes.size() / 2],
             gpu_sizes.back());
      // Show the smallest GPU group's P*M^2 vs the threshold
      printf("  [partition] SPLIT_COST auto-tuned from %zu groups, %d CPU + %d GPU threads\n",
             cubes.size() + spheres.size(), cpu_threads, gpu_threads);
    }
  }

  timeforgroup.resize(cubes.size() + spheres.size());
  next.resize(G2G::cpu_threads + G2G::gpu_threads);

  fort_forces_ms.resize(G2G::cpu_threads + G2G::gpu_threads);
  if (!fortran_vars.OPEN) {
    rmm_outputs.resize(G2G::cpu_threads + G2G::gpu_threads);
  } else {
    rmm_outputs_a.resize(G2G::cpu_threads + G2G::gpu_threads);
    rmm_outputs_b.resize(G2G::cpu_threads + G2G::gpu_threads);
  };

  for (int i = 0; i < G2G::cpu_threads + G2G::gpu_threads; i++) {
    fort_forces_ms[i].resize(fortran_vars.max_atoms, 3);
    if (fortran_vars.OPEN) {
      rmm_outputs_a[i].resize(fortran_vars.rmm_output_a.width,
                              fortran_vars.rmm_output_a.height);
      rmm_outputs_b[i].resize(fortran_vars.rmm_output_b.width,
                              fortran_vars.rmm_output_b.height);
    } else {
      rmm_outputs[i].resize(fortran_vars.rmm_output.width,
                            fortran_vars.rmm_output.height);
    }
  }
  int current_gpu = 0;
  for (int i = work.size(); i < G2G::cpu_threads + G2G::gpu_threads; i++)
    work.push_back(vector<int>());

  for (uint i = 0; i < cubes.size(); i++)
    if (cubes[i]->is_big_group()) {
      work[G2G::cpu_threads + current_gpu].push_back(i);
      current_gpu = (current_gpu + 1) % G2G::gpu_threads;
    }

  for (uint i = 0; i < spheres.size(); i++)
    if (spheres[i]->is_big_group()) {
      work[G2G::cpu_threads + current_gpu].push_back(i + cubes.size());
      current_gpu = (current_gpu + 1) % G2G::gpu_threads;
    }
  if (verbose > 4) diagnostic();
}
