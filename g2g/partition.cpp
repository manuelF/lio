#include <iostream>
#include <limits>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <climits>
#include "common.h"
#include "init.h"
#include "matrix.h"
#include "partition.h"
#include "partition_cost.h"
#include "timer.h"
using namespace std;

namespace G2G {

int MINCOST, THRESHOLD, SPLITPOINTS;
long long SPLIT_COST = 0;
GPUHardware gpu_hw = {0, 0, 0, 0, 0, false};
uint g2g_solve_epoch = 0;
Partition partition;

ostream& operator<<(ostream& io, const Timers& t) {
  ostringstream ss;
  ss << "density = " << t.density << " rmm = " << t.rmm
     << " forces = " << t.forces << " functions = " << t.functions;
  io << ss.str() << endl;
  return io;
}

/********************
 * PointGroup
 ********************/

template <class scalar_type>
void PointGroupCPU<scalar_type>::output_cost() const {
  // printf("%d %d %d %lld %lld\n", number_of_points, total_functions(),
  // rmm_bigs.size(), cost(), size_in_gpu());
}

template <class scalar_type>
bool PointGroupCPU<scalar_type>::is_big_group() const {
  return false;
}

#if GPU_KERNELS
template <class scalar_type>
bool PointGroupGPU<scalar_type>::is_big_group() const {
  return true;
}
#endif

template <class scalar_type>
void PointGroup<scalar_type>::move_base_from(PointGroup<scalar_type>& src) {
  points = std::move(src.points);
  number_of_points = src.number_of_points;
  s_functions = src.s_functions;
  p_functions = src.p_functions;
  d_functions = src.d_functions;
  func2global_nuc = std::move(src.func2global_nuc);
  func2local_nuc = std::move(src.func2local_nuc);
  local2global_func = std::move(src.local2global_func);
  local2global_nuc = std::move(src.local2global_nuc);
  inGlobal = src.inGlobal;
}

bool should_use_gpu(unsigned int points, unsigned int total_functions) {
  if (cpu_threads == 0) return true;
  if (gpu_threads == 0) return false;
  long long pm2 = (long long)points * total_functions * total_functions;
  return pm2 > SPLIT_COST;
}

// cores_per_sm, estimate_speed_ratio, compute_optimal_split_cost are now
// in partition_cost.{h,cpp}.

#if GPU_KERNELS
template <class scalar_type>
void PointGroupGPU<scalar_type>::get_rmm_input(
    HostMatrix<scalar_type>& rmm_input, FortranMatrix<double>& source) const {
  rmm_input.zero();
  // Use pre-computed index arrays (rmm_bigs/rmm_rows/rmm_cols from
  // compute_indexes). rmm_rows[k] <= rmm_cols[k] always (swap in
  // compute_indexes). Fill only the lower triangle for the density kernel:
  // HostMatrix operator()(i,j) = data[j*width+i], so first arg = column,
  // second = row. rmm_input(rows[k], cols[k]) gives col=rows[k] <= row=cols[k].
  const int indexes = static_cast<int>(this->rmm_bigs.size());
  for (int k = 0; k < indexes; k++) {
    scalar_type val = (scalar_type)source.data[this->rmm_bigs[k]];
    rmm_input(this->rmm_rows[k], this->rmm_cols[k]) = val;
  }
}
#endif

template <class scalar_type>
void PointGroupCPU<scalar_type>::get_rmm_input(
    HostMatrix<scalar_type>& rmm_input, FortranMatrix<double>& source) const {
  rmm_input.zero();
  const int indexes = static_cast<int>(this->rmm_bigs.size());
  for (int i = 0; i < indexes; i++) {
    int ii = this->rmm_rows[i], jj = this->rmm_cols[i], bi = this->rmm_bigs[i];
    rmm_input(ii, jj) = rmm_input(jj, ii) = (scalar_type)source.data[bi];
  }
}

template <class scalar_type>
void PointGroupCPU<scalar_type>::get_rmm_input(
    HostMatrix<scalar_type>& rmm_input) const {
  get_rmm_input(rmm_input, fortran_vars.rmm_input_ndens1);
}

#if GPU_KERNELS
template <class scalar_type>
void PointGroupGPU<scalar_type>::get_rmm_input(
    HostMatrix<scalar_type>& rmm_input) const {
  get_rmm_input(rmm_input, fortran_vars.rmm_input_ndens1);
}

template <class scalar_type>
void PointGroupGPU<scalar_type>::get_rmm_input(
    HostMatrix<scalar_type>& rmm_input_a,
    HostMatrix<scalar_type>& rmm_input_b) const {
  get_rmm_input(rmm_input_a, fortran_vars.rmm_dens_a);
  get_rmm_input(rmm_input_b, fortran_vars.rmm_dens_b);
}
#endif

template <class scalar_type>
void PointGroupCPU<scalar_type>::get_rmm_input(
    HostMatrix<scalar_type>& rmm_input_a,
    HostMatrix<scalar_type>& rmm_input_b) const {
  get_rmm_input(rmm_input_a, fortran_vars.rmm_dens_a);
  get_rmm_input(rmm_input_b, fortran_vars.rmm_dens_b);
}

template <class scalar_type>
void PointGroup<scalar_type>::compute_indexes() {
  rmm_bigs.clear();
  rmm_cols.clear();
  rmm_rows.clear();
  for (uint i = 0, ii = 0; i < this->total_functions_simple(); i++) {
    uint inc_i = this->small_function_type(i);

    for (uint k = 0; k < inc_i; k++, ii++) {
      uint big_i = this->local2global_func[i] + k;
      for (uint j = 0, jj = 0; j < this->total_functions_simple(); j++) {
        uint inc_j = this->small_function_type(j);

        for (uint l = 0; l < inc_j; l++, jj++) {
          uint big_j = this->local2global_func[j] + l;
          if (big_i > big_j) continue;
          uint big_index =
              (big_i * fortran_vars.m - (big_i * (big_i - 1)) / 2) +
              (big_j - big_i);
          if (ii > jj) swap(ii, jj);
          rmm_rows.push_back(ii);
          rmm_cols.push_back(jj);
          rmm_bigs.push_back(big_index);
        }
      }
    }
  }
}

template <class scalar_type>
void PointGroup<scalar_type>::add_rmm_output(
    const HostMatrix<scalar_type>& rmm_output,
    FortranMatrix<double>& target) const {
  for (uint i = 0, ii = 0; i < total_functions_simple(); i++) {
    uint inc_i = small_function_type(i);

    for (uint k = 0; k < inc_i; k++, ii++) {
      uint big_i = local2global_func[i] + k;
      for (uint j = 0, jj = 0; j < total_functions_simple(); j++) {
        uint inc_j = small_function_type(j);

        for (uint l = 0; l < inc_j; l++, jj++) {
          uint big_j = local2global_func[j] + l;
          if (big_i > big_j) continue;
          uint big_index =
              (big_i * fortran_vars.m - (big_i * (big_i - 1)) / 2) +
              (big_j - big_i);
          target(big_index) += (double)rmm_output(ii, jj);
        }
      }
    }
  }
}

template <class scalar_type>
void PointGroup<scalar_type>::add_rmm_output(
    const HostMatrix<scalar_type>& rmm_output,
    HostMatrix<double>& target) const {
  for (uint i = 0, ii = 0; i < total_functions_simple(); i++) {
    uint inc_i = small_function_type(i);

    for (uint k = 0; k < inc_i; k++, ii++) {
      uint big_i = local2global_func[i] + k;
      for (uint j = 0, jj = 0; j < total_functions_simple(); j++) {
        uint inc_j = small_function_type(j);

        for (uint l = 0; l < inc_j; l++, jj++) {
          uint big_j = local2global_func[j] + l;
          if (big_i > big_j) continue;
          uint big_index =
              (big_i * fortran_vars.m - (big_i * (big_i - 1)) / 2) +
              (big_j - big_i);
          target(big_index) += (double)rmm_output(ii, jj);
        }
      }
    }
  }
}

template <class scalar_type>
void PointGroup<scalar_type>::add_rmm_output(
    const HostMatrix<scalar_type>& rmm_output) const {
  add_rmm_output(rmm_output, fortran_vars.rmm_output);
}

template <class scalar_type>
void PointGroup<scalar_type>::add_rmm_output_a(
    const HostMatrix<scalar_type>& rmm_output) const {
  add_rmm_output(rmm_output, fortran_vars.rmm_output_a);
}

template <class scalar_type>
void PointGroup<scalar_type>::add_rmm_output_b(
    const HostMatrix<scalar_type>& rmm_output) const {
  add_rmm_output(rmm_output, fortran_vars.rmm_output_b);
}

template <class scalar_type>
void PointGroup<scalar_type>::add_rmm_open_output(
    const HostMatrix<scalar_type>& rmm_output_a,
    const HostMatrix<scalar_type>& rmm_output_b) const {
  add_rmm_output(rmm_output_a, fortran_vars.rmm_output_a);
  add_rmm_output(rmm_output_b, fortran_vars.rmm_output_b);
}

template <class scalar_type>
void PointGroup<scalar_type>::add_point(const Point& p) {
  points.push_back(p);
  number_of_points++;
}

#define EXP_PREFACTOR 1.01057089636005  // (2 * pow(4, 1/3.0)) / M_PI

template <class scalar_type>
bool PointGroup<scalar_type>::is_significative(FunctionType type,
                                               double exponent, double coeff,
                                               double d2) {
  switch (type) {
    case FUNCTION_S:
      return (exponent * d2 <
              max_function_exponent - log(pow((2. * exponent / M_PI), 3)) / 4);
      break;
    default: {
      double x = 1;
      double delta;
      double e = 0.1;
      double factor = pow((2.0 * exponent / M_PI), 3);
      factor = sqrt(factor * 4.0 * exponent);
      double norm = (type == FUNCTION_P ? sqrt(factor) : abs(factor));
      do {
        double div = (type == FUNCTION_P ? log(x) : 2 * log(x));
        double x1 = sqrt((max_function_exponent - log(norm) + div) / exponent);
        delta = abs(x - x1);
        x = x1;
      } while (delta > e);
      return (sqrt(d2) < x);
    } break;
  }
}

template <class scalar_type>
long long PointGroup<scalar_type>::cost() const {
  long long np = number_of_points, gm = total_functions();
  return 10 * ((np * gm * (1 + gm)) / 2) + MINCOST;
}

template <class scalar_type>
bool PointGroup<scalar_type>::operator<(
    const PointGroup<scalar_type>& T) const {
  return cost() < T.cost();
}

template <class scalar_type>
int PointGroup<scalar_type>::elements() const {
  int t = total_functions(), n = number_of_points;
  return t * n;
}

template <class scalar_type>
size_t PointGroup<scalar_type>::size_in_gpu() const {
  uint total_cost = 0;
  uint single_matrix_cost =
      COALESCED_DIMENSION(number_of_points) * total_functions();

  total_cost += single_matrix_cost * 2;  // 1 scalar_type functions + transposed
  if (fortran_vars.do_forces || fortran_vars.gga)
    total_cost += (single_matrix_cost * 4 * 2);  // 4 vec_type gradient + transposed
  if (fortran_vars.gga)
    total_cost += (single_matrix_cost * 8);  // 2*4 vec_type hessian
  return total_cost *
         sizeof(scalar_type);  // size in bytes according to precision
}

template<class scalar_type>
PointGroup<scalar_type>::~PointGroup<scalar_type>() {
}

template <class scalar_type>
PointGroupCPU<scalar_type>::~PointGroupCPU<scalar_type>() {
  deallocate();
}

template <class scalar_type>
void PointGroupCPU<scalar_type>::deallocate() {
  function_values.deallocate();
  gX.deallocate();
  gY.deallocate();
  gZ.deallocate();
  hPX.deallocate();
  hPY.deallocate();
  hPZ.deallocate();
  hIX.deallocate();
  hIY.deallocate();
  hIZ.deallocate();
  function_values_transposed.deallocate();
}

#if GPU_KERNELS
template <class scalar_type>
void PointGroupGPU<scalar_type>::deallocate() {
  if (this->inGlobal) {
    GlobalMemoryPool::dealloc(this->size_in_gpu(), current_device);
    function_values.deallocate();
    gradient_values.deallocate();
    hessian_values_transposed.deallocate();
    function_values_transposed.deallocate();
    gradient_values_transposed.deallocate();

    if (rmm_cuArray) {
      cudaDestroyTextureObject(rmm_tex);
      cudaFreeArray(rmm_cuArray);
      rmm_cuArray = nullptr;
      rmm_tex = 0;
    }

    if (rmm_cuArray_a) {
      cudaDestroyTextureObject(rmm_tex_a);
      cudaFreeArray(rmm_cuArray_a);
      rmm_cuArray_a = nullptr;
      rmm_tex_a = 0;
    }

    if (rmm_cuArray_b) {
      cudaDestroyTextureObject(rmm_tex_b);
      cudaFreeArray(rmm_cuArray_b);
      rmm_cuArray_b = nullptr;
      rmm_tex_b = 0;
    }

    rmm_input_gpu.deallocate();
    rmm_bigs_gpu.deallocate();
    rmm_rows_gpu.deallocate();
    rmm_cols_gpu.deallocate();

    // Deallocate cached temporary matrices
    partial_densities_gpu.deallocate();
    dxyz_gpu.deallocate();
    dd1_gpu.deallocate();
    dd2_gpu.deallocate();
    factors_gpu.deallocate();

    partial_densities_a_gpu.deallocate();
    dxyz_a_gpu.deallocate();
    dd1_a_gpu.deallocate();
    dd2_a_gpu.deallocate();

    partial_densities_b_gpu.deallocate();
    dxyz_b_gpu.deallocate();
    dd1_b_gpu.deallocate();
    dd2_b_gpu.deallocate();

    factors_a_gpu.deallocate();
    factors_b_gpu.deallocate();

    hessian_values.deallocate();
    rmm_output_gpu.deallocate();
    rmm_output_a_gpu.deallocate();
    rmm_output_b_gpu.deallocate();

    point_weights_gpu.deallocate();
    point_weights_cpu.deallocate();

    dd_gpu.deallocate();
    forces_gpu.deallocate();

    dd_gpu_a.deallocate();
    dd_gpu_b.deallocate();
    forces_gpu_a.deallocate();
    forces_gpu_b.deallocate();

    energy_host.deallocate();
    forces_host.deallocate();
    rmm_output_host.deallocate();

    energy_a_host.deallocate();
    energy_b_host.deallocate();
    forces_a_host.deallocate();
    forces_b_host.deallocate();

    accumulated_densities_gpu.deallocate();
    dxyz_accum_gpu.deallocate();
    dd1_accum_gpu.deallocate();
    dd2_accum_gpu.deallocate();

    this->inGlobal = false;
  }
}

template <class scalar_type>
PointGroupGPU<scalar_type>::~PointGroupGPU<scalar_type>() {
  deallocate();
  if (transpose_stream_1) {
    cudaStreamDestroy(transpose_stream_1);
    transpose_stream_1 = 0;
  }
  if (transpose_stream_2) {
    cudaStreamDestroy(transpose_stream_2);
    transpose_stream_2 = 0;
  }
}
#endif

void Partition::compute_functions(bool forces, bool gga) {
  Timer t1;
  t1.start();

#pragma omp parallel for schedule(guided, 8)
  for (uint i = 0; i < cubes.size(); i++) {
    if (!cubes[i]->is_big_group()) cubes[i]->compute_functions(forces, gga);
  }

#pragma omp parallel for schedule(guided, 8)
  for (uint i = 0; i < spheres.size(); i++) {
    if (!spheres[i]->is_big_group()) spheres[i]->compute_functions(forces, gga);
  }

  t1.stop();
  if (timer_single) cout << "Functions: " << t1 << endl;
}

void Partition::clear() {
  for (uint i = 0; i < cubes.size(); i++) delete cubes[i];
  for (uint i = 0; i < spheres.size(); i++) delete spheres[i];
  cubes.clear();
  spheres.clear();
  work.clear();
}

void Partition::rebalance(vector<double>& times, vector<double>& finishes) {
  for (int device = 0; device < 2; device++) {
    for (int rondas = 0; rondas < 5; rondas++) {
      int largest = 0;
      int smallest = 0;

      if (device == 0) {
        largest = static_cast<int>(
            std::max_element(finishes.begin(), finishes.end() - gpu_threads) -
            finishes.begin());
        smallest = static_cast<int>(
            std::min_element(finishes.begin(), finishes.end() - gpu_threads) -
            finishes.begin());
      } else {
        largest = static_cast<int>(
            std::max_element(finishes.begin() + cpu_threads, finishes.end()) -
            finishes.begin());
        smallest = static_cast<int>(
            std::min_element(finishes.begin() + cpu_threads, finishes.end()) -
            finishes.begin());
      }

      double diff = finishes[largest] - finishes[smallest];

      if (largest != smallest && work[largest].size() > 1) {
        double lt = finishes[largest];
        double moved = 0;
        while (diff / lt >= 0.02) {
          int mini = -1;
          double currentmini = diff;
          for (uint i = 0; i < work[largest].size(); i++) {
            int ind = work[largest][i];
            if (times[ind] > diff / 2) continue;
            double cost = times[ind];
            if (currentmini > diff - 2 * cost) {
              currentmini = diff - 2 * cost;
              mini = i;
            }
          }

          if (mini == -1) {
            // printf("Nothing more to swap!\n");
            break;
          }

          int topass = mini;
          int workindex = work[largest][topass];

          // printf("Swapping %d from %d to %d\n", work[largest][topass],
          // largest, smallest);

          if (device == 1) {
            if (workindex < (int)cubes.size())
              cubes[workindex]->deallocate();
            else
              spheres[workindex - cubes.size()]->deallocate();
          }

          work[smallest].push_back(work[largest][topass]);
          work[largest].erase(work[largest].begin() + topass);

          diff -= 2 * times[workindex];
          moved += times[workindex];
        }
        finishes[largest] -= moved;
        finishes[smallest] += moved;
      }
    }
  }
}

void Partition::solve(Timers& timers, bool compute_rmm, bool lda,
                      bool compute_forces, bool compute_energy,
                      double* fort_energy_ptr, double* fort_forces_ptr,
                      bool OPEN) {
  double energy = 0.0;

  double cubes_energy = 0, spheres_energy = 0;
  double cubes_energy_i = 0, spheres_energy_i = 0;
  double cubes_energy_c = 0, spheres_energy_c = 0;
  double cubes_energy_c1 = 0, spheres_energy_c1 = 0;
  double cubes_energy_c2 = 0, spheres_energy_c2 = 0;

  Timer smallgroups, biggroups;

  // Signal GPU groups that a new iteration has started, so shared device
  // buffers (global RMM) are re-uploaded exactly once.
  g2g_solve_epoch++;

// Verificar si anda reduction (+:energy) FF
#pragma omp parallel for num_threads(cpu_threads + gpu_threads) schedule( \
    static) reduction(+ : energy)
  for (uint i = 0; i < work.size(); i++) {
#if GPU_KERNELS
    bool gpu_thread = false;
    if (i >= cpu_threads) {
      gpu_thread = true;
      cudaSetDevice(i - cpu_threads);
    }
#endif
    double local_energy = 0;

    Timers ts;
    Timer t;
    t.start();

    if (compute_forces) fort_forces_ms[i].zero();
    if (compute_rmm) {
      if (OPEN) {
        rmm_outputs_a[i].zero();
        rmm_outputs_b[i].zero();
      } else {
        rmm_outputs[i].zero();
      }
    }

    for (uint j = 0; j < work[i].size(); j++) {
      int ind = work[i][j];
      Timer element;
      element.start();
      if (OPEN) {
        if (ind >= cubes.size()) {
          spheres[ind - cubes.size()]->solve_opened(
              ts, compute_rmm, lda, compute_forces, compute_energy,
              local_energy, spheres_energy_i, spheres_energy_c,
              spheres_energy_c1, spheres_energy_c2, fort_forces_ms[i],
              rmm_outputs_a[i], rmm_outputs_b[i]);
        } else {
          cubes[ind]->solve_opened(
              ts, compute_rmm, lda, compute_forces, compute_energy,
              local_energy, spheres_energy_i, spheres_energy_c,
              spheres_energy_c1, spheres_energy_c2, fort_forces_ms[i],
              rmm_outputs_a[i], rmm_outputs_b[i]);
        }
      } else {
        if (ind >= cubes.size()) {
          spheres[ind - cubes.size()]->solve_closed(
              ts, compute_rmm, lda, compute_forces, compute_energy,
              local_energy, fort_forces_ms[i], 1, rmm_outputs[i]);
        } else {
          cubes[ind]->solve_closed(ts, compute_rmm, lda, compute_forces,
                                   compute_energy, local_energy,
                                   fort_forces_ms[i], 1, rmm_outputs[i]);
        }
      }

      element.stop();
      timeforgroup[ind] = element.getTotal();
    }

#if GPU_KERNELS
    // After all GPU groups finish: sync and download the accumulated Fock
    // matrix from GPU into this thread's rmm_outputs buffer.
    if (gpu_thread && compute_rmm) {
      cudaStreamSynchronize(0);
      uint M = fortran_vars.m;
      uint rmm_global_size = M * (M + 1) / 2;
      if (OPEN) {
        download_gpu_fock_open(rmm_outputs_a[i].data, rmm_outputs_b[i].data,
                               rmm_global_size);
      } else {
        download_gpu_fock(rmm_outputs[i].data, rmm_global_size);
      }
    }
#endif

    t.stop();

    next[i] = t.getTotal();

    energy += local_energy;
  }

  Timer enditer;
  enditer.start();

  // Print CPU vs GPU thread balance per SCF iteration
  if (verbose > 3 && cpu_threads > 0 && gpu_threads > 0) {
    double max_cpu = 0.0;
    for (int i = 0; i < cpu_threads; i++)
      if (next[i] > max_cpu) max_cpu = next[i];
    double gpu_time = next[cpu_threads];
    int cpu_groups = 0, gpu_groups = 0;
    for (int i = 0; i < cpu_threads; i++) cpu_groups += static_cast<int>(work[i].size());
    for (int i = cpu_threads; i < cpu_threads + gpu_threads; i++)
      gpu_groups += static_cast<int>(work[i].size());
    printf("  [balance] CPU max=%.1fms (%d groups/%d threads)  "
           "GPU=%.1fms (%d groups/%d threads)  "
           "idle=%.1fms (%s waits)\n",
           max_cpu / 1e6, cpu_groups, cpu_threads,
           gpu_time / 1e6, gpu_groups, gpu_threads,
           fabs(gpu_time - max_cpu) / 1e6,
           gpu_time > max_cpu ? "CPU" : "GPU");
  }

  // Dump per-group timing data on the second SCF iteration (first is cold)
  static int solve_call = 0;
  solve_call++;
  if (verbose > 3 && solve_call == 2) {
    printf("  [perfmodel] idx,device,points,functions,cost,time_us\n");
    for (uint i = 0; i < cubes.size(); i++) {
      printf("  [perfmodel] %u,%s,%u,%u,%lld,%.1f\n",
             i, cubes[i]->is_big_group() ? "GPU" : "CPU",
             cubes[i]->number_of_points, cubes[i]->total_functions(),
             cubes[i]->cost(), timeforgroup[i] / 1e3);
    }
    for (uint i = 0; i < spheres.size(); i++) {
      uint idx = i + static_cast<uint>(cubes.size());
      printf("  [perfmodel] %u,%s,%u,%u,%lld,%.1f\n",
             idx, spheres[i]->is_big_group() ? "GPU" : "CPU",
             spheres[i]->number_of_points, spheres[i]->total_functions(),
             spheres[i]->cost(), timeforgroup[idx] / 1e3);
    }
  }

  if (work.size() > 1) rebalance(timeforgroup, next);
  if (compute_forces) {
    FortranMatrix<double> fort_forces_out(fort_forces_ptr, fortran_vars.atoms,
                                          3, fortran_vars.max_atoms);
    // Kahan compensated summation reduces FP rounding noise from per-thread
    // accumulation order changes caused by rebalance().
    const int force_elems = fortran_vars.atoms * 3;
    std::vector<double> force_comp(force_elems, 0.0);
    for (uint k = 0; k < fort_forces_ms.size(); k++) {
      for (int i = 0; i < fortran_vars.atoms; i++) {
        for (int j = 0; j < 3; j++) {
          int idx = i * 3 + j;
          double y = fort_forces_ms[k](i, j) - force_comp[idx];
          double t = fort_forces_out(i, j) + y;
          force_comp[idx] = (t - fort_forces_out(i, j)) - y;
          fort_forces_out(i, j) = t;
        }
      }
    }
  }

  if (compute_rmm) {
    if (fortran_vars.OPEN) {
      double* dst_a = fortran_vars.rmm_output_a.data;
      double* dst_b = fortran_vars.rmm_output_b.data;
      const int elements =
          fortran_vars.rmm_output_a.width * fortran_vars.rmm_output_a.height;
      const int elements_b =
          fortran_vars.rmm_output_b.width * fortran_vars.rmm_output_b.height;

      if (!(rmm_outputs_a.size() == rmm_outputs_b.size())) {
        std::cout << "ERROR in partition.solve: outputs A and B of different "
                     "size. \n";
      }

      if (!(elements == elements_b)) {
        std::cout << "ERROR in partition.solve: different number of elements A "
                     "and B.\n";
      }

      std::vector<double> comp_a(elements, 0.0);
      std::vector<double> comp_b(elements, 0.0);
      for (uint k = 0; k < rmm_outputs_a.size(); k++) {
        const double* src_a = rmm_outputs_a[k].asArray();
        const double* src_b = rmm_outputs_b[k].asArray();
        for (int i = 0; i < elements; i++) {
          double y_a = src_a[i] - comp_a[i];
          double t_a = dst_a[i] + y_a;
          comp_a[i] = (t_a - dst_a[i]) - y_a;
          dst_a[i] = t_a;

          double y_b = src_b[i] - comp_b[i];
          double t_b = dst_b[i] + y_b;
          comp_b[i] = (t_b - dst_b[i]) - y_b;
          dst_b[i] = t_b;
        }
      }
    } else {
      double* dst = fortran_vars.rmm_output.data;
      const int elements =
          fortran_vars.rmm_output.width * fortran_vars.rmm_output.height;
      std::vector<double> comp(elements, 0.0);
      for (uint k = 0; k < rmm_outputs.size(); k++) {
        const double* src = rmm_outputs[k].asArray();
        for (int i = 0; i < elements; i++) {
          double y = src[i] - comp[i];
          double t = dst[i] + y;
          comp[i] = (t - dst[i]) - y;
          dst[i] = t;
        }
      }
    }
  }

/*  if (OPEN && compute_energy) {
    std::cout << " Ei:  " << cubes_energy_i + spheres_energy_i << std::endl;
    std::cout << " Ec:  " << cubes_energy_c + spheres_energy_c << std::endl;
    std::cout << " Ec1: " << cubes_energy_c1 + spheres_energy_c1 << std::endl;
    std::cout << " Ec2: " << cubes_energy_c2 + spheres_energy_c2 << std::endl;
  }*/

  *fort_energy_ptr = energy;
  if (*fort_energy_ptr != *fort_energy_ptr) {
    std::cout << "I see dead peaple " << std::endl;
#if GPU_KERNELS
    cudaDeviceReset();
#endif
    exit(1);
  }
}

#if FULL_DOUBLE
template class PointGroup<double>;
template class PointGroupCPU<double>;
#if GPU_KERNELS
template class PointGroupGPU<double>;
#endif
#else
template class PointGroup<float>;
template class PointGroupCPU<float>;
#if GPU_KERNELS
template class PointGroupGPU<float>;
#endif
#endif
}
