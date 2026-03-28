/* -*- mode: c -*- */
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <math_constants.h>
#include <string>
#include <vector>

#include "../common.h"
#include "../init.h"
#include "cuda_extra.h"
#include "../matrix.h"
#include "../timer.h"
#include "../partition.h"
#include "../scalar_vector_types.h"
#include "../global_memory_pool.h"

#include "../pointxc/calc_ggaCS.h"
#include "../pointxc/calc_ggaOS.h"
#include "../pointxc/calc_ldaCS.h"

#if USE_LIBXC
#include "../libxc/libxc_accumulate_point.h"
#endif

namespace G2G {
/** KERNELS **/
#include "gpu_variables.h"
#include "kernels/accumulate_point.h"
#include "kernels/energy.h"
#include "kernels/energy_open.h"
#include "kernels/energy_derivs.h"
#include "kernels/rmm.h"
#include "kernels/weight.h"
#include "kernels/functions.h"
#include "kernels/force.h"
#include "kernels/transpose.h"
#include "kernels/rmm_gather.h"
#include "kernels/rmm_scatter.h"

using std::cout;
using std::endl;
using std::vector;

// Static GPU buffers for GPU-side Fock scatter (shared across all GPU groups).
// Each group's gpu_scatter_rmm atomicAdds into these; downloaded once per
// iteration after all GPU groups finish.
static CudaMatrix<double> s_global_fock_dev;
static CudaMatrix<double> s_global_fock_a_dev;
static CudaMatrix<double> s_global_fock_b_dev;
static uint s_fock_epoch = 0;

void download_gpu_fock(double* output, uint n_elements) {
  if (s_global_fock_dev.is_allocated()) {
    cudaMemcpy(output, s_global_fock_dev.data,
               n_elements * sizeof(double), cudaMemcpyDeviceToHost);
  }
}

void download_gpu_fock_open(double* output_a, double* output_b,
                            uint n_elements) {
  if (s_global_fock_a_dev.is_allocated()) {
    cudaMemcpy(output_a, s_global_fock_a_dev.data,
               n_elements * sizeof(double), cudaMemcpyDeviceToHost);
  }
  if (s_global_fock_b_dev.is_allocated()) {
    cudaMemcpy(output_b, s_global_fock_b_dev.data,
               n_elements * sizeof(double), cudaMemcpyDeviceToHost);
  }
}

// extern "C" void g2g_timer_sum_start_(const char* timer_name, unsigned int
// length_arg); extern "C" void g2g_timer_sum_stop_(const char* timer_name,
// unsigned int length_arg); extern "C" void g2g_timer_sum_pause_(const char*
// timer_name, unsigned int length_arg);

void gpu_set_variables(void) {
  int previous_device;
  cudaGetDevice(&previous_device);
  int gpu_devices = cudaGetGPUCount();
  for (int i = 0; i < gpu_devices; i++) {
    if (cudaSetDevice(i) != cudaSuccess)
      std::cout << "Error: can't set the device " << i << std::endl;
    cudaMemcpyToSymbol(
        gpu_normalization_factor, &fortran_vars.normalization_factor,
        sizeof(fortran_vars.normalization_factor), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_atoms, &fortran_vars.atoms,
                       sizeof(fortran_vars.atoms), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_Iexch, &fortran_vars.iexch,
                       sizeof(fortran_vars.iexch), 0, cudaMemcpyHostToDevice);
  }
  cudaSetDevice(previous_device);
  cudaAssertNoError("set_gpu_variables");
}

template <class T>
void gpu_set_atom_positions(const HostMatrix<T>& m) {
  int previous_device;
  cudaGetDevice(&previous_device);
  int gpu_devices = cudaGetGPUCount();
  for (int i = 0; i < gpu_devices; i++) {
    if (cudaSetDevice(i) != cudaSuccess)
      std::cout << "Error: can't set the device " << i << std::endl;
    cudaMemcpyToSymbol(gpu_atom_positions, m.data, m.bytes(), 0,
                       cudaMemcpyHostToDevice);
  }
  cudaSetDevice(previous_device);
}

#if FULL_DOUBLE
template void gpu_set_atom_positions<double3>(const HostMatrix<double3>& m);
#else
template void gpu_set_atom_positions<float3>(const HostMatrix<float3>& m);
#endif

template <class scalar_type>
void PointGroupGPU<scalar_type>::solve(
    Timers& timers, bool compute_rmm, bool lda, bool compute_forces,
    bool compute_energy, double& energy, double& energy_i, double& energy_c,
    double& energy_c1, double& energy_c2, HostMatrix<double>& fort_forces_ms,
    int inner_threads, HostMatrix<double>& rmm_output_local, bool open) {
  /*
    if ( open ) {
        solve_opened( timers, compute_rmm, lda, compute_forces, compute_energy,
                      energy, energy_i, energy_c, energy_c1, energy_c2,
                      fort_forces_ms );
    }
    else {
        solve_closed( timers, compute_rmm, lda, compute_forces, compute_energy,
                      energy, fort_forces_ms, inner_threads, rmm_output_local );
    }
  */
  //  counter_iter++; // For Debug FF std::cout << "Grupo " << counter_iter << "
  //  Energia : " << energy << " \n"; // For Debug FF
}

template <class scalar_type>
void PointGroupGPU<scalar_type>::solve_closed(
    Timers& timers, bool compute_rmm, bool lda, bool compute_forces,
    bool compute_energy, double& energy, HostMatrix<double>& fort_forces_ms,
    int inner_threads, HostMatrix<double>& rmm_output_local) {
  int device;
  cudaGetDevice(&device);
  current_device = device;

  /*** Computo sobre cada cubo ****/

  /** Compute this group's functions **/
  timers.functions.start();
  compute_functions(compute_forces, !lda);
  timers.functions.pause();

  uint group_m = this->total_functions();

  timers.density.start();
  /** Load points from group **/
  if (!point_weights_gpu.is_allocated() || !this->inGlobal) {
    point_weights_cpu.resize(this->number_of_points, 1);

    uint i = 0;
    for (vector<Point>::const_iterator p = this->points.begin();
         p != this->points.end(); ++p, ++i) {
      point_weights_cpu(i) = p->weight;
    }

    point_weights_gpu = point_weights_cpu;
  }

  dim3 threadBlock, threadGrid;
  /* compute density/factors */

  const int block_height = divUp(group_m, 2 * DENSITY_BLOCK_SIZE);

  threadBlock =
      dim3(DENSITY_BLOCK_SIZE, 1,
           1);  // Hay que asegurarse que la cantidad de funciones este en rango
  threadGrid = dim3(this->number_of_points, block_height, 1);

  partial_densities_gpu.resize(COALESCED_DIMENSION(this->number_of_points),
                               block_height);
  dxyz_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dd1_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dd2_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);

#if USE_LIBXC
  accumulated_densities_gpu.resize(COALESCED_DIMENSION(this->number_of_points),
                                   block_height);
  dxyz_accum_gpu.resize(COALESCED_DIMENSION(this->number_of_points),
                        block_height);
  dd1_accum_gpu.resize(COALESCED_DIMENSION(this->number_of_points),
                       block_height);
  dd2_accum_gpu.resize(COALESCED_DIMENSION(this->number_of_points),
                       block_height);
#endif

  // TODO: que libxc_gpu reciba estos datos para los kernels, asi todos usan lo
  // mismo.
  const dim3 threadGrid_accumulate(
      divUp(this->number_of_points, DENSITY_ACCUM_BLOCK_SIZE), 1, 1);
  const dim3 threadBlock_accumulate(DENSITY_ACCUM_BLOCK_SIZE, 1, 1);

  if (compute_rmm || compute_forces) {
    factors_gpu.resize(this->number_of_points);
    factors_gpu.zero();
  }

  int transposed_width = COALESCED_DIMENSION(this->number_of_points);
#define BLOCK_DIM 16
  dim3 transpose_grid(transposed_width / BLOCK_DIM, divUp((group_m), BLOCK_DIM),
                      1);
  dim3 transpose_threads(BLOCK_DIM, BLOCK_DIM, 1);

  /*
   **********************************************************************
   * GPU gather of group-local RMM from global packed-triangular RMM,
   * then D2D copy to CUDA array for texture reads.
   **********************************************************************
   */

  uint rmm_width = COALESCED_DIMENSION(group_m);
  uint rmm_height = group_m + DENSITY_BLOCK_SIZE;

  // Upload index arrays to GPU (once per group lifetime, cached)
  if (!rmm_bigs_gpu.is_allocated()) {
    uint n_idx = this->rmm_bigs.size();
    rmm_bigs_gpu.resize(n_idx, 1);
    rmm_rows_gpu.resize(n_idx, 1);
    rmm_cols_gpu.resize(n_idx, 1);
    cudaMemcpy(rmm_bigs_gpu.data, this->rmm_bigs.data(),
               n_idx * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(rmm_rows_gpu.data, this->rmm_rows.data(),
               n_idx * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(rmm_cols_gpu.data, this->rmm_cols.data(),
               n_idx * sizeof(uint), cudaMemcpyHostToDevice);
  }

  // Upload global packed RMM to GPU once per SCF iteration (shared across all
  // groups via static buffer; M*(M+1)/2 doubles ≈ 517 KB for M=364)
  {
    static CudaMatrix<double> s_global_rmm_dev;
    static uint s_last_epoch = 0;
    uint M = fortran_vars.m;
    uint rmm_global_size = M * (M + 1) / 2;
    if (!s_global_rmm_dev.is_allocated() ||
        s_global_rmm_dev.width != rmm_global_size) {
      s_global_rmm_dev.resize(rmm_global_size, 1);
    }
    if (s_last_epoch != g2g_solve_epoch) {
      cudaMemcpy(s_global_rmm_dev.data,
                 fortran_vars.rmm_input_ndens1.data,
                 rmm_global_size * sizeof(double), cudaMemcpyHostToDevice);
      s_last_epoch = g2g_solve_epoch;
    }

    // Allocate flat GPU buffer for gathered local RMM (cached per-group)
    rmm_input_gpu.resize(rmm_width, rmm_height);
    rmm_input_gpu.zero();

    // GPU gather: fill both triangles of local RMM
    uint n_indexes = this->rmm_bigs.size();
    dim3 gather_block(256);
    dim3 gather_grid((n_indexes + 255) / 256);
    gpu_gather_rmm<scalar_type><<<gather_grid, gather_block>>>(
        s_global_rmm_dev.data, rmm_bigs_gpu.data, rmm_rows_gpu.data,
        rmm_cols_gpu.data, rmm_input_gpu.data, n_indexes, rmm_width);
  }

  // Create texture array + object (one-time)
  if (rmm_cuArray == nullptr) {
    cudaChannelFormatDesc channelDesc =
#if FULL_DOUBLE
        cudaCreateChannelDesc<int2>();
#else
        cudaCreateChannelDesc<float>();
#endif
    cudaMallocArray(&rmm_cuArray, &channelDesc, rmm_width, rmm_height);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = rmm_cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaCreateTextureObject(&rmm_tex, &resDesc, &texDesc, NULL);
  }

  // D2D copy from flat GPU buffer to CUDA array for texture reads
  cudaMemcpy2DToArrayAsync(rmm_cuArray, 0, 0, rmm_input_gpu.data,
                           rmm_width * sizeof(scalar_type),
                           rmm_width * sizeof(scalar_type),
                           rmm_height, cudaMemcpyDeviceToDevice);

  cudaTextureObject_t rmm_input_gpu_tex = rmm_tex;

#if USE_LIBXC
  const int nspin = XC_UNPOLARIZED;
  const int functionalExchange = fortran_vars.ex_functional_id;     // 1101;
  const int functionalCorrelation = fortran_vars.ec_functional_id;  // 1130;
  LibxcProxy<scalar_type, 4> libxcProxy;
  if (fortran_vars.use_libxc) {
    libxcProxy.init(functionalExchange, functionalCorrelation, nspin);
  }
#endif

  if (compute_energy) {
    energy_gpu.resize(this->number_of_points);

#define compute_parameters                                                 \
  rmm_input_gpu_tex, energy_gpu.data, factors_gpu.data,                    \
      point_weights_gpu.data, this->number_of_points,                      \
      function_values_transposed.data, gradient_values_transposed.data,    \
      hessian_values_transposed.data, group_m, partial_densities_gpu.data, \
      dxyz_gpu.data, dd1_gpu.data, dd2_gpu.data

#define accumulate_parameters                                           \
  energy_gpu.data, factors_gpu.data, point_weights_gpu.data,            \
      this->number_of_points, block_height, partial_densities_gpu.data, \
      dxyz_gpu.data, dd1_gpu.data, dd2_gpu.data

    // VER QUE PASA SI SACAMOS COMPUTE_FACTOR Y COMPUTE ENERGY DE
    // gpu_compute_density
    if (compute_forces || compute_rmm) {
      if (lda) {
        gpu_compute_density<scalar_type, true>
            <<<threadGrid, threadBlock>>>(compute_parameters);
        gpu_accumulate_point<scalar_type, true, true, true>
            <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                accumulate_parameters);
      } else {
        gpu_compute_density<scalar_type, false>
            <<<threadGrid, threadBlock>>>(compute_parameters);
#if USE_LIBXC
        if (fortran_vars.use_libxc) {
          // Accumulate the data for libxc
          gpu_accumulate_point_for_libxc<scalar_type, true, true, false>
              <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                  point_weights_gpu.data, this->number_of_points, block_height,
                  partial_densities_gpu.data, dxyz_gpu.data, dd1_gpu.data,
                  dd2_gpu.data, accumulated_densities_gpu.data,
                  dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);
#if LIBXC_CPU
          // Compute exc_corr and y2a with libxc CPU version.
          libxc_exchange_correlation_cpu<scalar_type, true, true, false>(
              &libxcProxy, energy_gpu.data, factors_gpu.data,
              this->number_of_points, accumulated_densities_gpu.data,
              dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);
#else
          // Compute exc_corr and y2a with libxc GPU version.
          libxc_exchange_correlation_gpu<scalar_type, true, true, false>(
              &libxcProxy, energy_gpu.data, factors_gpu.data,
              this->number_of_points, accumulated_densities_gpu.data,
              dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);
#endif
          // Merge the results.
          gpu_accumulate_energy_and_forces_from_libxc<scalar_type, true, true,
                                                      false>
              <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                  energy_gpu.data, factors_gpu.data, point_weights_gpu.data,
                  this->number_of_points, accumulated_densities_gpu.data);
        } else {
          gpu_accumulate_point<scalar_type, true, true, false>
              <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                  accumulate_parameters);
        }
#else
        // print_accumulate_parameters<scalar_type> (accumulate_parameters);
        gpu_accumulate_point<scalar_type, true, true, false>
            <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                accumulate_parameters);
#endif
      }
    } else {
      if (lda) {
        gpu_compute_density<scalar_type, true>
            <<<threadGrid, threadBlock>>>(compute_parameters);
        gpu_accumulate_point<scalar_type, true, false, true>
            <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                accumulate_parameters);
      } else {
        gpu_compute_density<scalar_type, false>
            <<<threadGrid, threadBlock>>>(compute_parameters);
#if USE_LIBXC
        if (fortran_vars.use_libxc) {
          // Accumulate the data.
          gpu_accumulate_point_for_libxc<scalar_type, true, false, false>
              <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                  point_weights_gpu.data, this->number_of_points, block_height,
                  partial_densities_gpu.data, dxyz_gpu.data, dd1_gpu.data,
                  dd2_gpu.data, accumulated_densities_gpu.data,
                  dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);

#if LIBXC_CPU
          // Compute exc_corr and y2a with CPU libxc.
          libxc_exchange_correlation_cpu<scalar_type, true, false, false>(
              &libxcProxy, energy_gpu.data, factors_gpu.data,
              this->number_of_points, accumulated_densities_gpu.data,
              dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);
#else
          // Compute exc_corr and y2a with libxc GPU version.
          libxc_exchange_correlation_gpu<scalar_type, true, true, false>(
              &libxcProxy, energy_gpu.data, factors_gpu.data,
              this->number_of_points, accumulated_densities_gpu.data,
              dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);
#endif
          // Merge the results.
          gpu_accumulate_energy_and_forces_from_libxc<scalar_type, true, true,
                                                      false>
              <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                  energy_gpu.data, factors_gpu.data, point_weights_gpu.data,
                  this->number_of_points, accumulated_densities_gpu.data);

        } else {
          gpu_accumulate_point<scalar_type, true, false, false>
              <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                  accumulate_parameters);
        }
#else
        gpu_accumulate_point<scalar_type, true, false, false>
            <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                accumulate_parameters);
#endif
      }
    }
    cudaAssertNoError("compute_density");

    energy_host.resize(this->number_of_points);
    energy_host.copy_submatrix_async(energy_gpu, 0);
    cudaStreamSynchronize(0);
    for (uint i = 0; i < this->number_of_points; i++) {
      energy += (double)energy_host(i);
    }
  } else {
#undef compute_parameters
#undef accumulate_parameters

#define compute_parameters                                              \
  rmm_input_gpu_tex, NULL, factors_gpu.data, point_weights_gpu.data,    \
      this->number_of_points, function_values_transposed.data,          \
      gradient_values_transposed.data, hessian_values_transposed.data,  \
      group_m, partial_densities_gpu.data, dxyz_gpu.data, dd1_gpu.data, \
      dd2_gpu.data
#define accumulate_parameters                                                \
  NULL, factors_gpu.data, point_weights_gpu.data, this->number_of_points,    \
      block_height, partial_densities_gpu.data, dxyz_gpu.data, dd1_gpu.data, \
      dd2_gpu.data
    if (lda) {
      gpu_compute_density<scalar_type, true>
          <<<threadGrid, threadBlock>>>(compute_parameters);
      gpu_accumulate_point<scalar_type, false, true, true>
          <<<threadGrid_accumulate, threadBlock_accumulate>>>(
              accumulate_parameters);
    } else {
      gpu_compute_density<scalar_type, false>
          <<<threadGrid, threadBlock>>>(compute_parameters);
#if USE_LIBXC
      if (fortran_vars.use_libxc) {
        // Accumulate the data.
        gpu_accumulate_point_for_libxc<scalar_type, false, true, false>
            <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                point_weights_gpu.data, this->number_of_points, block_height,
                partial_densities_gpu.data, dxyz_gpu.data, dd1_gpu.data,
                dd2_gpu.data, accumulated_densities_gpu.data,
                dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);

#if LIBXC_CPU
        // Compute exc_corr and y2a with libxc CPU.
        libxc_exchange_correlation_cpu<scalar_type, false, true, false>(
            &libxcProxy, NULL, factors_gpu.data, this->number_of_points,
            accumulated_densities_gpu.data, dxyz_accum_gpu.data,
            dd1_accum_gpu.data, dd2_accum_gpu.data);
#else
        // Compute exc_corr and y2a with libxc GPU version.
        libxc_exchange_correlation_gpu<scalar_type, false, true, false>(
            &libxcProxy, NULL, factors_gpu.data, this->number_of_points,
            accumulated_densities_gpu.data, dxyz_accum_gpu.data,
            dd1_accum_gpu.data, dd2_accum_gpu.data);
#endif
        // Merge the results.
        gpu_accumulate_energy_and_forces_from_libxc<scalar_type, false, true,
                                                    false>
            <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                NULL, factors_gpu.data, point_weights_gpu.data,
                this->number_of_points, accumulated_densities_gpu.data);
      } else {
        gpu_accumulate_point<scalar_type, false, true, false>
            <<<threadGrid_accumulate, threadBlock_accumulate>>>(
                accumulate_parameters);
      }
#else
      gpu_accumulate_point<scalar_type, false, true, false>
          <<<threadGrid_accumulate, threadBlock_accumulate>>>(
              accumulate_parameters);
#endif
    }
    cudaAssertNoError("compute_density");
  }
#undef compute_parameters
#undef accumulate_parameters

  timers.density.pause();
  /* compute forces */
  if (compute_forces) {
    // gpu_gather_rmm already filled both triangles of the texture, so no
    // CPU symmetrize or re-upload needed for density_derivs.

    timers.density_derivs.start();
    dim3 threads = dim3(this->number_of_points);
    threadBlock = dim3(DENSITY_DERIV_BLOCK_SIZE);
    threadGrid = divUp(threads, threadBlock);

    dd_gpu.resize(COALESCED_DIMENSION(this->number_of_points),
                  this->total_nucleii());
    dd_gpu.zero();
    CudaMatrixUInt nuc_gpu(
        this->func2local_nuc);  // TODO: esto en realidad se podria guardar una
                                // sola vez durante su construccion

    gpu_compute_density_derivs<<<threadGrid, threadBlock>>>(
        rmm_input_gpu_tex, function_values.data, gradient_values.data,
        nuc_gpu.data, dd_gpu.data, this->number_of_points, group_m,
        this->total_nucleii());
    cudaAssertNoError("density_derivs");
    timers.density_derivs.pause();

    timers.forces.start();
    forces_gpu.resize(this->total_nucleii());
    forces_gpu.zero();

    threads = dim3(this->total_nucleii());
    threadBlock = dim3(FORCE_BLOCK_SIZE);
    threadGrid = divUp(threads, threadBlock);
    gpu_compute_forces<<<threadGrid, threadBlock>>>(
        this->number_of_points, factors_gpu.data, dd_gpu.data, forces_gpu.data,
        this->total_nucleii());
    cudaAssertNoError("forces");

    forces_host.resize(this->total_nucleii());
    forces_host.copy_submatrix_async(forces_gpu, 0);
    cudaStreamSynchronize(0);

    for (uint i = 0; i < this->total_nucleii(); ++i) {
      vec_type4 atom_force = forces_host(i);
      uint global_nuc = this->local2global_nuc[i];
      fort_forces_ms(global_nuc, 0) += (double)atom_force.x;
      fort_forces_ms(global_nuc, 1) += (double)atom_force.y;
      fort_forces_ms(global_nuc, 2) += (double)atom_force.z;
    }
    timers.forces.pause();
  }

  timers.rmm.start();
  /* compute RMM */
  if (compute_rmm) {
    threadBlock = dim3(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
    uint blocksPerRow = divUp(group_m, RMM_BLOCK_SIZE_XY);
    // Only use enough blocks for lower triangle
    threadGrid = dim3(blocksPerRow * (blocksPerRow + 1) / 2);

    rmm_output_gpu.resize(COALESCED_DIMENSION(group_m), group_m);
    rmm_output_gpu.zero();
    // For calls with a single block (pretty common with cubes) don't bother
    // doing the arithmetic to get block position in the matrix
    if (blocksPerRow > 1) {
      gpu_update_rmm<scalar_type, true><<<threadGrid, threadBlock>>>(
          factors_gpu.data, this->number_of_points, rmm_output_gpu.data,
          function_values.data, group_m);
    } else {
      gpu_update_rmm<scalar_type, false><<<threadGrid, threadBlock>>>(
          factors_gpu.data, this->number_of_points, rmm_output_gpu.data,
          function_values.data, group_m);
    }
    cudaAssertNoError("update_rmm");

    /*** Scatter local Fock to global packed Fock on GPU ***/
    {
      uint M = fortran_vars.m;
      uint rmm_global_size = M * (M + 1) / 2;
      if (!s_global_fock_dev.is_allocated() ||
          s_global_fock_dev.width != rmm_global_size) {
        s_global_fock_dev.resize(rmm_global_size, 1);
      }
      if (s_fock_epoch != g2g_solve_epoch) {
        cudaMemset(s_global_fock_dev.data, 0,
                   rmm_global_size * sizeof(double));
        s_fock_epoch = g2g_solve_epoch;
      }
      uint n_indexes = this->rmm_bigs.size();
      dim3 scatter_block(256);
      dim3 scatter_grid((n_indexes + 255) / 256);
      gpu_scatter_rmm<scalar_type><<<scatter_grid, scatter_block>>>(
          rmm_output_gpu.data, rmm_bigs_gpu.data, rmm_rows_gpu.data,
          rmm_cols_gpu.data, s_global_fock_dev.data, n_indexes, rmm_width);
    }
  }
  timers.rmm.pause();

  /* clear functions */
  if (!(this->inGlobal)) {
    function_values.deallocate();
    gradient_values.deallocate();
    hessian_values_transposed.deallocate();
    function_values_transposed.deallocate();
    gradient_values_transposed.deallocate();
  }
}

//======================
// OPENSHELL
//======================

template <class scalar_type>
void PointGroupGPU<scalar_type>::solve_opened(
    Timers& timers, bool compute_rmm, bool lda, bool compute_forces,
    bool compute_energy, double& energy, double& energy_i, double& energy_c,
    double& energy_c1, double& energy_c2, HostMatrix<double>& fort_forces_ms,
    HostMatrix<double>& rmm_output_local_a,
    HostMatrix<double>& rmm_output_local_b) {
  int device;
  cudaGetDevice(&device);
  current_device = device;

  /*** Computo sobre cada cubo ****/

  /** Compute this group's functions **/
  timers.functions.start();
  compute_functions(compute_forces, !lda);
  timers.functions.pause();

  uint group_m = this->total_functions();

  timers.density.start();
  /** Load points from group **/
  if (!point_weights_gpu.is_allocated() || !this->inGlobal) {
    point_weights_cpu.resize(this->number_of_points, 1);

    uint i = 0;
    for (vector<Point>::const_iterator p = this->points.begin();
         p != this->points.end(); ++p, ++i) {
      point_weights_cpu(i) = p->weight;
    }
    point_weights_gpu = point_weights_cpu;
  }

  dim3 threadBlock, threadGrid;
  const int block_height = divUp(group_m, 2 * DENSITY_BLOCK_SIZE);

  // This makes sure the amount of functions fits within range.
  threadBlock = dim3(DENSITY_BLOCK_SIZE, 1, 1);
  threadGrid = dim3(this->number_of_points, block_height, 1);

  // Matrix transpose is needed for better coalescence in density.

  int transposed_width = COALESCED_DIMENSION(this->number_of_points);

#define BLOCK_DIM 16
  dim3 transpose_grid(transposed_width / BLOCK_DIM,
                      divUp((group_m), BLOCK_DIM));
  dim3 transpose_threads(BLOCK_DIM, BLOCK_DIM, 1);

  partial_densities_a_gpu.resize(COALESCED_DIMENSION(this->number_of_points),
                                 block_height);
  dxyz_a_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dd1_a_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dd2_a_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);

  partial_densities_b_gpu.resize(COALESCED_DIMENSION(this->number_of_points),
                                 block_height);
  dxyz_b_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dd1_b_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dd2_b_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);

  const dim3 threadGrid_accumulate(
      divUp(this->number_of_points, DENSITY_ACCUM_BLOCK_SIZE), 1, 1);
  const dim3 threadBlock_accumulate(DENSITY_ACCUM_BLOCK_SIZE, 1, 1);

  if (compute_rmm || compute_forces) {
    factors_a_gpu.resize(this->number_of_points);
    factors_b_gpu.resize(this->number_of_points);
    factors_a_gpu.zero();
    factors_b_gpu.zero();
  }

  /*
   **********************************************************************
   * GPU gather of alpha/beta RMM from global packed RMM,
   * then D2D copy to CUDA arrays for texture reads.
   **********************************************************************
   */

  uint rmm_width = COALESCED_DIMENSION(group_m);
  uint rmm_height = group_m + DENSITY_BLOCK_SIZE;

  // Upload index arrays to GPU (once per group lifetime, cached)
  if (!rmm_bigs_gpu.is_allocated()) {
    uint n_idx = this->rmm_bigs.size();
    rmm_bigs_gpu.resize(n_idx, 1);
    rmm_rows_gpu.resize(n_idx, 1);
    rmm_cols_gpu.resize(n_idx, 1);
    cudaMemcpy(rmm_bigs_gpu.data, this->rmm_bigs.data(),
               n_idx * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(rmm_rows_gpu.data, this->rmm_rows.data(),
               n_idx * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(rmm_cols_gpu.data, this->rmm_cols.data(),
               n_idx * sizeof(uint), cudaMemcpyHostToDevice);
  }

  // GPU gather for alpha and beta density matrices (upload once per iteration)
  {
    static CudaMatrix<double> s_global_rmm_a_dev;
    static CudaMatrix<double> s_global_rmm_b_dev;
    static uint s_last_epoch_open = 0;
    uint M = fortran_vars.m;
    uint rmm_global_size = M * (M + 1) / 2;

    if (!s_global_rmm_a_dev.is_allocated() ||
        s_global_rmm_a_dev.width != rmm_global_size) {
      s_global_rmm_a_dev.resize(rmm_global_size, 1);
      s_global_rmm_b_dev.resize(rmm_global_size, 1);
    }
    if (s_last_epoch_open != g2g_solve_epoch) {
      cudaMemcpy(s_global_rmm_a_dev.data, fortran_vars.rmm_dens_a.data,
                 rmm_global_size * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(s_global_rmm_b_dev.data, fortran_vars.rmm_dens_b.data,
                 rmm_global_size * sizeof(double), cudaMemcpyHostToDevice);
      s_last_epoch_open = g2g_solve_epoch;
    }

    // Allocate flat GPU buffers for gathered local RMM (reuse rmm_input_gpu
    // for alpha; allocate separate buffer for beta)
    // resize() is a no-op if dimensions match; must always call because
    // the static beta buffer is shared across groups with different group_m.
    rmm_input_gpu.resize(rmm_width, rmm_height);
    static CudaMatrix<scalar_type> rmm_input_b_gpu_local;
    rmm_input_b_gpu_local.resize(rmm_width, rmm_height);
    rmm_input_gpu.zero();
    rmm_input_b_gpu_local.zero();

    uint n_indexes = this->rmm_bigs.size();
    dim3 gather_block(256);
    dim3 gather_grid((n_indexes + 255) / 256);

    // Gather alpha
    gpu_gather_rmm<scalar_type><<<gather_grid, gather_block>>>(
        s_global_rmm_a_dev.data, rmm_bigs_gpu.data, rmm_rows_gpu.data,
        rmm_cols_gpu.data, rmm_input_gpu.data, n_indexes, rmm_width);
    // Gather beta
    gpu_gather_rmm<scalar_type><<<gather_grid, gather_block>>>(
        s_global_rmm_b_dev.data, rmm_bigs_gpu.data, rmm_rows_gpu.data,
        rmm_cols_gpu.data, rmm_input_b_gpu_local.data, n_indexes, rmm_width);

    // Create texture arrays + objects (one-time)
    if (rmm_cuArray_a == nullptr) {
      cudaChannelFormatDesc channelDesc =
#if FULL_DOUBLE
          cudaCreateChannelDesc<int2>();
#else
          cudaCreateChannelDesc<float>();
#endif
      cudaMallocArray(&rmm_cuArray_a, &channelDesc, rmm_width, rmm_height);
      cudaMallocArray(&rmm_cuArray_b, &channelDesc, rmm_width, rmm_height);

      cudaResourceDesc resDesc1 = {};
      resDesc1.resType = cudaResourceTypeArray;
      resDesc1.res.array.array = rmm_cuArray_a;

      cudaResourceDesc resDesc2 = {};
      resDesc2.resType = cudaResourceTypeArray;
      resDesc2.res.array.array = rmm_cuArray_b;

      cudaTextureDesc texDesc = {};
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModePoint;
      texDesc.readMode = cudaReadModeElementType;
      texDesc.normalizedCoords = 0;

      cudaCreateTextureObject(&rmm_tex_a, &resDesc1, &texDesc, NULL);
      cudaCreateTextureObject(&rmm_tex_b, &resDesc2, &texDesc, NULL);
    }

    // D2D copy from flat GPU buffers to CUDA arrays
    cudaMemcpy2DToArrayAsync(rmm_cuArray_a, 0, 0, rmm_input_gpu.data,
                             rmm_width * sizeof(scalar_type),
                             rmm_width * sizeof(scalar_type),
                             rmm_height, cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArrayAsync(rmm_cuArray_b, 0, 0,
                             rmm_input_b_gpu_local.data,
                             rmm_width * sizeof(scalar_type),
                             rmm_width * sizeof(scalar_type),
                             rmm_height, cudaMemcpyDeviceToDevice);
  }

  cudaTextureObject_t rmm_input_gpu_tex = rmm_tex_a;
  cudaTextureObject_t rmm_input_gpu_tex2 = rmm_tex_b;

  if (compute_energy) {
    energy_gpu.resize(this->number_of_points);
    energy_i_gpu.resize(this->number_of_points);
    energy_c_gpu.resize(this->number_of_points);
    energy_c1_gpu.resize(this->number_of_points);
    energy_c2_gpu.resize(this->number_of_points);

    if (compute_forces || compute_rmm) {
      gpu_compute_density_opened<scalar_type, true, true, false>
          <<<threadGrid, threadBlock>>>(
              rmm_input_gpu_tex, rmm_input_gpu_tex2, point_weights_gpu.data,
              this->number_of_points, function_values_transposed.data,
              gradient_values_transposed.data, hessian_values_transposed.data,
              group_m, partial_densities_a_gpu.data, dxyz_a_gpu.data,
              dd1_a_gpu.data, dd2_a_gpu.data, partial_densities_b_gpu.data,
              dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
      gpu_accumulate_point_open<scalar_type, true, true, false>
          <<<threadGrid_accumulate, threadBlock_accumulate>>>(
              energy_gpu.data, energy_i_gpu.data, energy_c_gpu.data,
              energy_c1_gpu.data, energy_c2_gpu.data, factors_a_gpu.data,
              factors_b_gpu.data, point_weights_gpu.data,
              this->number_of_points, block_height,
              partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data,
              dd2_a_gpu.data, partial_densities_b_gpu.data, dxyz_b_gpu.data,
              dd1_b_gpu.data, dd2_b_gpu.data);
    } else {
      gpu_compute_density_opened<scalar_type, true, false, false>
          <<<threadGrid, threadBlock>>>(
              rmm_input_gpu_tex, rmm_input_gpu_tex2, point_weights_gpu.data,
              this->number_of_points, function_values_transposed.data,
              gradient_values_transposed.data, hessian_values_transposed.data,
              group_m, partial_densities_a_gpu.data, dxyz_a_gpu.data,
              dd1_a_gpu.data, dd2_a_gpu.data, partial_densities_b_gpu.data,
              dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
      gpu_accumulate_point_open<scalar_type, true, false, false>
          <<<threadGrid_accumulate, threadBlock_accumulate>>>(
              energy_gpu.data, energy_i_gpu.data, energy_c_gpu.data,
              energy_c1_gpu.data, energy_c2_gpu.data, factors_a_gpu.data,
              factors_b_gpu.data, point_weights_gpu.data,
              this->number_of_points, block_height,
              partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data,
              dd2_a_gpu.data, partial_densities_b_gpu.data, dxyz_b_gpu.data,
              dd1_b_gpu.data, dd2_b_gpu.data);
    }
    cudaAssertNoError("compute_density");

    HostMatrix<scalar_type> energy_cpu(energy_gpu);
    HostMatrix<scalar_type> energy_i_cpu(energy_i_gpu);
    HostMatrix<scalar_type> energy_c_cpu(energy_c_gpu);
    HostMatrix<scalar_type> energy_c1_cpu(energy_c1_gpu);
    HostMatrix<scalar_type> energy_c2_cpu(energy_c2_gpu);

    for (uint i = 0; i < this->number_of_points; i++) {
      energy += (double)energy_cpu(i);
      energy_i += (double)energy_i_cpu(i);
      energy_c += (double)energy_c_cpu(i);
      energy_c1 += (double)energy_c1_cpu(i);
      energy_c2 += (double)energy_c2_cpu(i);
    }
  } else {
    gpu_compute_density_opened<scalar_type, false, true, false>
        <<<threadGrid, threadBlock>>>(
            rmm_input_gpu_tex, rmm_input_gpu_tex2, point_weights_gpu.data,
            this->number_of_points, function_values_transposed.data,
            gradient_values_transposed.data, hessian_values_transposed.data,
            group_m, partial_densities_a_gpu.data, dxyz_a_gpu.data,
            dd1_a_gpu.data, dd2_a_gpu.data, partial_densities_b_gpu.data,
            dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
    gpu_accumulate_point_open<scalar_type, false, true, false>
        <<<threadGrid_accumulate, threadBlock_accumulate>>>(
            NULL, NULL, NULL, NULL, NULL, factors_a_gpu.data,
            factors_b_gpu.data, point_weights_gpu.data, this->number_of_points,
            block_height, partial_densities_a_gpu.data, dxyz_a_gpu.data,
            dd1_a_gpu.data, dd2_a_gpu.data, partial_densities_b_gpu.data,
            dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
    cudaAssertNoError("compute_density");
  }

  timers.density.pause();

  /* compute forces */
  if (compute_forces) {
    // gpu_gather_rmm already filled both triangles of the textures, so no
    // CPU symmetrize or re-upload needed for density_derivs_open.

    dim3 threads;
    timers.density_derivs.start();
    threads = dim3(this->number_of_points);
    threadBlock = dim3(DENSITY_DERIV_BLOCK_SIZE);
    threadGrid = divUp(threads, threadBlock);

    dd_gpu_a.resize(COALESCED_DIMENSION(this->number_of_points),
                    this->total_nucleii());
    dd_gpu_b.resize(COALESCED_DIMENSION(this->number_of_points),
                    this->total_nucleii());
    dd_gpu_a.zero();
    dd_gpu_b.zero();
    CudaMatrixUInt nuc_gpu(this->func2local_nuc);

    // Kernel
    gpu_compute_density_derivs_open<<<threadGrid, threadBlock>>>(
        rmm_input_gpu_tex, rmm_input_gpu_tex2, function_values.data,
        gradient_values.data, nuc_gpu.data, dd_gpu_a.data, dd_gpu_b.data,
        this->number_of_points, group_m, this->total_nucleii());

    cudaAssertNoError("density_derivs");
    timers.density_derivs.pause();

    timers.forces.start();
    forces_gpu_a.resize(this->total_nucleii());
    forces_gpu_b.resize(this->total_nucleii());
    forces_gpu_a.zero();
    forces_gpu_b.zero();

    threads = dim3(this->total_nucleii());
    threadBlock = dim3(FORCE_BLOCK_SIZE);
    threadGrid = divUp(threads, threadBlock);
    // Kernel
    gpu_compute_forces<<<threadGrid, threadBlock>>>(
        this->number_of_points, factors_a_gpu.data, dd_gpu_a.data,
        forces_gpu_a.data, this->total_nucleii());
    gpu_compute_forces<<<threadGrid, threadBlock>>>(
        this->number_of_points, factors_b_gpu.data, dd_gpu_b.data,
        forces_gpu_b.data, this->total_nucleii());

    cudaAssertNoError("forces");

    HostMatrix<vec_type4> forces_cpu_a(forces_gpu_a);
    HostMatrix<vec_type4> forces_cpu_b(forces_gpu_b);

    for (uint i = 0; i < this->total_nucleii(); ++i) {
      vec_type4 atom_force_a = forces_cpu_a(i);
      vec_type4 atom_force_b = forces_cpu_b(i);
      uint global_nuc = this->local2global_nuc[i];

      fort_forces_ms(global_nuc, 0) += (double)atom_force_a.x + (double)atom_force_b.x;
      fort_forces_ms(global_nuc, 1) += (double)atom_force_a.y + (double)atom_force_b.y;
      fort_forces_ms(global_nuc, 2) += (double)atom_force_a.z + (double)atom_force_b.z;
    }

    timers.forces.pause();
  }

  /* compute RMM */
  timers.rmm.start();
  if (compute_rmm) {
    threadBlock = dim3(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
    uint blocksPerRow = divUp(group_m, RMM_BLOCK_SIZE_XY);
    // Only use enough blocks for lower triangle
    threadGrid = dim3(blocksPerRow * (blocksPerRow + 1) / 2);

    rmm_output_a_gpu.resize(COALESCED_DIMENSION(group_m), group_m);
    rmm_output_b_gpu.resize(COALESCED_DIMENSION(group_m), group_m);
    rmm_output_a_gpu.zero();
    rmm_output_b_gpu.zero();
    // For calls with a single block (pretty common with cubes) don't bother
    // doing the arithmetic to get block position in the matrix
    if (blocksPerRow > 1) {
      gpu_update_rmm<scalar_type, true><<<threadGrid, threadBlock>>>(
          factors_a_gpu.data, this->number_of_points, rmm_output_a_gpu.data,
          function_values.data, group_m);
      gpu_update_rmm<scalar_type, true><<<threadGrid, threadBlock>>>(
          factors_b_gpu.data, this->number_of_points, rmm_output_b_gpu.data,
          function_values.data, group_m);
    } else {
      gpu_update_rmm<scalar_type, false><<<threadGrid, threadBlock>>>(
          factors_a_gpu.data, this->number_of_points, rmm_output_a_gpu.data,
          function_values.data, group_m);
      gpu_update_rmm<scalar_type, false><<<threadGrid, threadBlock>>>(
          factors_b_gpu.data, this->number_of_points, rmm_output_b_gpu.data,
          function_values.data, group_m);
    }
    cudaAssertNoError("update_rmm");
    /*** Scatter local Fock (alpha+beta) to global packed Fock on GPU ***/
    {
      uint M = fortran_vars.m;
      uint rmm_global_size = M * (M + 1) / 2;
      if (!s_global_fock_a_dev.is_allocated() ||
          s_global_fock_a_dev.width != rmm_global_size) {
        s_global_fock_a_dev.resize(rmm_global_size, 1);
        s_global_fock_b_dev.resize(rmm_global_size, 1);
      }
      if (s_fock_epoch != g2g_solve_epoch) {
        cudaMemset(s_global_fock_a_dev.data, 0,
                   rmm_global_size * sizeof(double));
        cudaMemset(s_global_fock_b_dev.data, 0,
                   rmm_global_size * sizeof(double));
        s_fock_epoch = g2g_solve_epoch;
      }
      uint n_indexes = this->rmm_bigs.size();
      dim3 scatter_block(256);
      dim3 scatter_grid((n_indexes + 255) / 256);
      gpu_scatter_rmm<scalar_type><<<scatter_grid, scatter_block>>>(
          rmm_output_a_gpu.data, rmm_bigs_gpu.data, rmm_rows_gpu.data,
          rmm_cols_gpu.data, s_global_fock_a_dev.data, n_indexes, rmm_width);
      gpu_scatter_rmm<scalar_type><<<scatter_grid, scatter_block>>>(
          rmm_output_b_gpu.data, rmm_bigs_gpu.data, rmm_rows_gpu.data,
          rmm_cols_gpu.data, s_global_fock_b_dev.data, n_indexes, rmm_width);
    }
  }
  timers.rmm.pause();

  /* clear functions */
  if (!(this->inGlobal)) {
    function_values.deallocate();
    gradient_values.deallocate();
    hessian_values_transposed.deallocate();
    function_values_transposed.deallocate();
    gradient_values_transposed.deallocate();
  }

  // uint free_memory, total_memory;
  // cudaGetMemoryInfo(free_memory, total_memory);
  // cout << "Maximum used memory: " << (double)max_used_memory / (1024 * 1024)
  // << "MB (" << ((double)max_used_memory / total_memory) * 100.0 << "%)" <<
  // endl; cudaPrintMemoryInfo();
}

/*******************************
 * Cube Functions
 *******************************/

template <class scalar_type>
void PointGroupGPU<scalar_type>::compute_functions(bool forces, bool gga) {
  if (this->inGlobal) {  // Ya las tengo en memoria? entonces salgo porque ya
                         // estan las 3 calculadas
    return;
  }

  if (0 == GlobalMemoryPool::tryAlloc(
               this->size_in_gpu()))  // 1 si hubo error, 0 si pude reservar la
                                      // memoria
    this->inGlobal = true;
  CudaMatrix<vec_type4> points_position_gpu;
  CudaMatrix<vec_type2> factor_ac_gpu;
  CudaMatrixUInt nuc_gpu;
  CudaMatrixUInt contractions_gpu;

  /** Load points from group **/
  {
    HostMatrix<vec_type4> points_position_cpu(this->number_of_points, 1);
    uint i = 0;
    for (vector<Point>::const_iterator p = this->points.begin();
         p != this->points.end(); ++p, ++i) {
      points_position_cpu(i) =
          vec_type4(p->position.x, p->position.y, p->position.z, 0);
    }
    points_position_gpu = points_position_cpu;
  }
  /* Load group functions */
  uint group_m =
      this->s_functions + this->p_functions * 3 + this->d_functions * 6;
  uint4 group_functions = make_uint4(this->s_functions, this->p_functions,
                                     this->d_functions, group_m);
  HostMatrix<vec_type2> factor_ac_cpu(COALESCED_DIMENSION(group_m),
                                      MAX_CONTRACTIONS);
  HostMatrixUInt nuc_cpu(group_m, 1), contractions_cpu(group_m, 1);

  // TODO: hacer que functions.h itere por total_small_functions()... asi puedo
  // hacer que func2global_nuc sea de tamaño total_functions() y directamente
  // copio esa matriz aca y en otros lados

  uint ii = 0;
  for (uint i = 0; i < this->total_functions_simple(); ++i) {
    uint inc = this->small_function_type(i);

    uint func = this->local2global_func[i];
    uint this_nuc = this->func2global_nuc(i);
    uint this_cont = fortran_vars.contractions(func);

    for (uint j = 0; j < inc; j++) {
      nuc_cpu(ii) = this_nuc;
      contractions_cpu(ii) = this_cont;
      for (unsigned int k = 0; k < this_cont; k++)
        factor_ac_cpu(ii, k) = vec_type2(fortran_vars.a_values(func, k),
                                         fortran_vars.c_values(func, k));
      ii++;
    }
  }
  factor_ac_gpu = factor_ac_cpu;
  nuc_gpu = nuc_cpu;
  contractions_gpu = contractions_cpu;

  /** Compute Functions **/
  function_values.resize(COALESCED_DIMENSION(this->number_of_points),
                         group_functions.w);
  function_values_transposed.resize(
      group_m, COALESCED_DIMENSION(this->number_of_points));

  if (fortran_vars.do_forces || fortran_vars.gga) {
    gradient_values.resize(COALESCED_DIMENSION(this->number_of_points),
                           group_functions.w);
    gradient_values_transposed.resize(
        group_m, COALESCED_DIMENSION(this->number_of_points));
  }

  if (fortran_vars.gga)
    hessian_values.resize(COALESCED_DIMENSION(this->number_of_points),
                          (group_functions.w) * 2);

  dim3 threads(this->number_of_points);
  dim3 threadBlock(FUNCTIONS_BLOCK_SIZE);
  dim3 threadGrid = divUp(threads, threadBlock);

#define compute_functions_parameters                                       \
  points_position_gpu.data, this->number_of_points, contractions_gpu.data, \
      factor_ac_gpu.data, nuc_gpu.data, function_values.data,              \
      gradient_values.data, hessian_values.data, group_functions
  if (forces) {
    if (gga)
      gpu_compute_functions<scalar_type, true, true>
          <<<threadGrid, threadBlock>>>(compute_functions_parameters);
    else
      gpu_compute_functions<scalar_type, true, false>
          <<<threadGrid, threadBlock>>>(compute_functions_parameters);
  } else {
    if (gga)
      gpu_compute_functions<scalar_type, false, true>
          <<<threadGrid, threadBlock>>>(compute_functions_parameters);
    else
      gpu_compute_functions<scalar_type, false, false>
          <<<threadGrid, threadBlock>>>(compute_functions_parameters);
  }

#define TILE_DIM 32
#define BLOCK_ROWS 8
  int width = COALESCED_DIMENSION(this->number_of_points);
  int height = group_m;
  dim3 transpose_threads(TILE_DIM, BLOCK_ROWS, 1);
  dim3 transpose_grid(divUp(width, TILE_DIM), divUp(height, TILE_DIM), 1);

  // Lazy-init persistent transpose streams (blocking by default, so the
  // default/legacy stream automatically waits for them before its next op).
  // This eliminates ~3800 cudaStreamCreate/Destroy calls per run and allows
  // get_rmm_input CPU work to overlap with transpose GPU work:
  //   - compute_functions() returns without syncing
  //   - caller does get_rmm_input (CPU-only, no GPU dependency)
  //   - caller calls cudaMemcpy2DToArrayAsync on default stream, which CUDA
  //     serialises after all blocking streams (transpose_stream_1/2) complete
  if (!transpose_stream_1) cudaStreamCreate(&transpose_stream_1);
  if (!transpose_stream_2) cudaStreamCreate(&transpose_stream_2);

  transpose<<<transpose_grid, transpose_threads, 0, transpose_stream_1>>>(
      function_values_transposed.data, function_values.data,
      COALESCED_DIMENSION(this->number_of_points), group_m);

  if (fortran_vars.do_forces || fortran_vars.gga) {
    transpose<<<transpose_grid, transpose_threads, 0, transpose_stream_2>>>(
        gradient_values_transposed.data, gradient_values.data,
        COALESCED_DIMENSION(this->number_of_points), group_m);
  }

  if (fortran_vars.gga) {
    dim3 transpose_grid_hess(divUp(width, TILE_DIM),
                             divUp(height * 2, TILE_DIM), 1);
    hessian_values_transposed.resize(height * 2, width);

    transpose<<<transpose_grid_hess, transpose_threads, 0, transpose_stream_1>>>(
        hessian_values_transposed.data, hessian_values.data, width, height * 2);
  }
  // No explicit sync here — the caller's next default-stream operation
  // (cudaMemcpy2DToArrayAsync) implicitly waits for these blocking streams.

  cudaAssertNoError("compute_functions");
}

/*******************************
 * Cube Weights
 *******************************/
template <class scalar_type>
void PointGroupGPU<scalar_type>::compute_weights(void) {
  CudaMatrix<vec_type4> point_positions_gpu;
  CudaMatrix<vec_type4> atom_position_rm_gpu;
  {
    HostMatrix<vec_type4> points_positions_cpu(this->number_of_points, 1);
    uint i = 0;
    for (vector<Point>::const_iterator p = this->points.begin();
         p != this->points.end(); ++p, ++i) {
      points_positions_cpu(i) =
          vec_type4(p->position.x, p->position.y, p->position.z, p->atom);
    }
    point_positions_gpu = points_positions_cpu;

    HostMatrix<vec_type4> atom_position_rm_cpu(fortran_vars.atoms, 1);
    for (uint i = 0; i < fortran_vars.atoms; i++) {
      double3 atom_pos = fortran_vars.atom_positions(i);
      atom_position_rm_cpu(i) =
          vec_type4(atom_pos.x, atom_pos.y, atom_pos.z, fortran_vars.rm(i));
    }
    atom_position_rm_gpu = atom_position_rm_cpu;
  }
  CudaMatrixUInt nucleii_gpu(this->local2global_nuc);

  CudaMatrix<scalar_type>& weights_gpu = point_weights_gpu;
  weights_gpu.resize(this->number_of_points);
  dim3 threads(this->number_of_points);
  dim3 blockSize(WEIGHT_BLOCK_SIZE);
  dim3 gridSize = divUp(threads, blockSize);
  gpu_compute_weights<scalar_type><<<gridSize, blockSize>>>(
      this->number_of_points, point_positions_gpu.data,
      atom_position_rm_gpu.data, weights_gpu.data, nucleii_gpu.data,
      this->total_nucleii());
  cudaAssertNoError("compute_weights");

  point_weights_cpu = weights_gpu;
  weights_gpu.deallocate();  // force solve_closed() to upload full weights (quadrature*Becke)
  uint i = 0;
  for (vector<Point>::iterator p = this->points.begin();
       p != this->points.end(); ++p, ++i) {
    p->weight *= (double)point_weights_cpu(i);
  }
}

#if FULL_DOUBLE
template class PointGroup<double>;
template class PointGroupGPU<double>;
#else
template class PointGroup<float>;
template class PointGroupGPU<float>;
#endif

}  // namespace G2G
