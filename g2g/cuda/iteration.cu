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
#include "kernels/becke.h"
#include "kernels/rmm_scatter.h"

using std::cout;
using std::vector;
using std::endl;

// GPU-side packed Fock buffers shared across all GPU groups within a single
// Partition::solve() call. Each PointGroupGPU::solve_* atomicAdds its local
// scatter into these; partition.cpp downloads them once after the parallel
// region. Allocated lazily; zeroed by gpu_scatter_zero_global_fock.
static CudaMatrix<double> s_global_fock_dev;
static CudaMatrix<double> s_global_fock_a_dev;
static CudaMatrix<double> s_global_fock_b_dev;

void gpu_scatter_zero_global_fock(unsigned int rmm_global_size, bool open) {
  if (!open) {
    if (!s_global_fock_dev.is_allocated() ||
        s_global_fock_dev.width != rmm_global_size) {
      s_global_fock_dev.resize(rmm_global_size, 1);
    }
    cudaMemsetAsync(s_global_fock_dev.data, 0,
                    rmm_global_size * sizeof(double), 0);
  } else {
    if (!s_global_fock_a_dev.is_allocated() ||
        s_global_fock_a_dev.width != rmm_global_size) {
      s_global_fock_a_dev.resize(rmm_global_size, 1);
      s_global_fock_b_dev.resize(rmm_global_size, 1);
    }
    cudaMemsetAsync(s_global_fock_a_dev.data, 0,
                    rmm_global_size * sizeof(double), 0);
    cudaMemsetAsync(s_global_fock_b_dev.data, 0,
                    rmm_global_size * sizeof(double), 0);
  }
}

void gpu_scatter_download_global_fock(double* host_dst,
                                      unsigned int rmm_global_size) {
  if (!s_global_fock_dev.is_allocated()) return;
  cudaMemcpy(host_dst, s_global_fock_dev.data,
             rmm_global_size * sizeof(double), cudaMemcpyDeviceToHost);
}

// Free the per-PointGroupGPU cuArray + cudaTextureObject cache. Called from
// PointGroupGPU::deallocate() in partition.cpp via an opaque interface so
// partition.cpp does not need to include CUDA runtime headers.
void gpu_release_group_rmm_texture(void*& cuArray_ptr,
                                   unsigned long long& tex_handle) {
  if (tex_handle) {
    cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(tex_handle));
    tex_handle = 0;
  }
  if (cuArray_ptr) {
    cudaFreeArray(reinterpret_cast<cudaArray*>(cuArray_ptr));
    cuArray_ptr = NULL;
  }
}

void gpu_scatter_download_global_fock_open(double* host_dst_a,
                                           double* host_dst_b,
                                           unsigned int rmm_global_size) {
  if (s_global_fock_a_dev.is_allocated()) {
    cudaMemcpy(host_dst_a, s_global_fock_a_dev.data,
               rmm_global_size * sizeof(double), cudaMemcpyDeviceToHost);
  }
  if (s_global_fock_b_dev.is_allocated()) {
    cudaMemcpy(host_dst_b, s_global_fock_b_dev.data,
               rmm_global_size * sizeof(double), cudaMemcpyDeviceToHost);
  }
}

//extern "C" void g2g_timer_sum_start_(const char* timer_name, unsigned int length_arg);
//extern "C" void g2g_timer_sum_stop_(const char* timer_name, unsigned int length_arg);
//extern "C" void g2g_timer_sum_pause_(const char* timer_name, unsigned int length_arg);

void gpu_set_variables(void) {
  int previous_device; cudaGetDevice(&previous_device);
  int gpu_devices = cudaGetGPUCount();
  for(int i = 0; i < gpu_devices; i++) {
    if(cudaSetDevice(i) != cudaSuccess)
      std::cout << "Error: can't set the device " << i << std::endl;
    cudaMemcpyToSymbol(gpu_normalization_factor, &fortran_vars.normalization_factor, sizeof(fortran_vars.normalization_factor), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_atoms, &fortran_vars.atoms, sizeof(fortran_vars.atoms), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_Iexch, &fortran_vars.iexch, sizeof(fortran_vars.iexch), 0, cudaMemcpyHostToDevice);
  }
  cudaSetDevice(previous_device);
  cudaAssertNoError("set_gpu_variables");
}

template<class T> void gpu_set_atom_positions(const HostMatrix<T>& m) {
  int previous_device; cudaGetDevice(&previous_device);
  int gpu_devices = cudaGetGPUCount();
  for(int i = 0; i < gpu_devices; i++) {
    if(cudaSetDevice(i) != cudaSuccess)
      std::cout << "Error: can't set the device " << i << std::endl;
    cudaMemcpyToSymbol(gpu_atom_positions, m.data, m.bytes(), 0, cudaMemcpyHostToDevice);
  }
  cudaSetDevice(previous_device);
}

#if FULL_DOUBLE
template void gpu_set_atom_positions<double3>(const HostMatrix<double3>& m);
#else
template void gpu_set_atom_positions<float3>(const HostMatrix<float3>& m);
#endif

template<class scalar_type>
void PointGroupGPU<scalar_type>::solve(
    Timers& timers, bool compute_rmm, bool lda, bool compute_forces,
    bool compute_energy, double& energy,double& energy_i, double& energy_c,
    double& energy_c1, double& energy_c2,  HostMatrix<double>& fort_forces_ms,
    int inner_threads, HostMatrix<double>& rmm_output_local, bool open){
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
//  counter_iter++;                                                            // For Debug FF
//  std::cout << "Grupo " << counter_iter << " Energia : " << energy << " \n"; // For Debug FF
}

template<class scalar_type>
void PointGroupGPU<scalar_type>::solve_closed(
    Timers& timers,
    bool compute_rmm, bool lda, bool compute_forces, bool compute_energy,
    double& energy,    HostMatrix<double>& fort_forces_ms,
    int inner_threads, HostMatrix<double>& rmm_output_local,
    HostMatrix<double>& becke_dens, CDFTVars& my_cdft_vars){

  int device;
  cudaGetDevice(&device);
  current_device = device;

  /*** Computo sobre cada cubo ****/
  /** Compute this group's functions **/
  timers.functions.start_and_sync();
  compute_functions(compute_forces, !lda);
  timers.functions.pause_and_sync();

  uint group_m = this->total_functions();

  timers.density.start_and_sync();
  /** Load points from group (one-time upload — points[].weight is constant
   ** across all SCF iterations). **/
  if (!this->point_weights_gpu_cached.is_allocated() ||
      this->point_weights_gpu_cached.width != this->number_of_points) {
    HostMatrix<scalar_type> point_weights_cpu(this->number_of_points, 1);
    for (uint pi = 0; pi < this->number_of_points; ++pi) {
      point_weights_cpu(pi) = this->points[pi].weight;
    }
    this->point_weights_gpu_cached = point_weights_cpu;
  }
  CudaMatrix<scalar_type>& point_weights_gpu = this->point_weights_gpu_cached;

  dim3 threadBlock, threadGrid;
  /* compute density/factors */

  const int block_height= divUp(group_m, 2*DENSITY_BLOCK_SIZE);

  threadBlock = dim3(DENSITY_BLOCK_SIZE,1,1); // Hay que asegurarse que la cantidad de funciones este en rango
  threadGrid = dim3(this->number_of_points,block_height,1);

  CudaMatrix<scalar_type> partial_densities_gpu;
  CudaMatrix< vec_type<scalar_type,4> > dxyz_gpu;
  CudaMatrix< vec_type<scalar_type,4> > dd1_gpu;
  CudaMatrix< vec_type<scalar_type,4> > dd2_gpu;

  partial_densities_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dxyz_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height);
  dd1_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height );
  dd2_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height );

#if USE_LIBXC
  CudaMatrix<scalar_type> accumulated_densities_gpu;
  CudaMatrix< vec_type<scalar_type,4> > dxyz_accum_gpu;
  CudaMatrix< vec_type<scalar_type,4> > dd1_accum_gpu;
  CudaMatrix< vec_type<scalar_type,4> > dd2_accum_gpu;

  accumulated_densities_gpu.resize(COALESCED_DIMENSION(this->number_of_points));
  dxyz_accum_gpu.resize(COALESCED_DIMENSION(this->number_of_points));
  dd1_accum_gpu.resize(COALESCED_DIMENSION(this->number_of_points));
  dd2_accum_gpu.resize(COALESCED_DIMENSION(this->number_of_points));
#endif

  //TODO: que libxc_gpu reciba estos datos para los kernels, asi todos usan lo mismo.
  const dim3 threadGrid_accumulate(divUp(this->number_of_points,DENSITY_ACCUM_BLOCK_SIZE),1,1);
  const dim3 threadBlock_accumulate(DENSITY_ACCUM_BLOCK_SIZE,1,1);

  CudaMatrix<scalar_type> factors_gpu;
  if (compute_rmm || compute_forces) {
    factors_gpu.resize(this->number_of_points);
    factors_gpu.zero();
  }

  int transposed_width = COALESCED_DIMENSION(this->number_of_points);
  #define BLOCK_DIM 16
  dim3 transpose_grid(transposed_width / BLOCK_DIM, divUp((group_m),BLOCK_DIM), 1);
  dim3 transpose_threads(BLOCK_DIM, BLOCK_DIM, 1);

  // Reuse the per-group cached transposed buffers populated once in
  // compute_functions() when inGlobal. Otherwise (group too large for the
  // cache budget) fall back to a per-iter local transpose.
  CudaMatrix<scalar_type> function_values_transposed_local;
  CudaMatrix<vec_type<scalar_type,4> > gradient_values_transposed_local;
  scalar_type*   function_values_transposed_ptr;
  vec_type<scalar_type,4>* gradient_values_transposed_ptr = NULL;

  if (this->inGlobal && this->function_values_transposed_cached.is_allocated()) {
    function_values_transposed_ptr = this->function_values_transposed_cached.data;
    if (fortran_vars.do_forces || fortran_vars.gga) {
      gradient_values_transposed_ptr = this->gradient_values_transposed_cached.data;
    }
  } else {
    function_values_transposed_local.resize(group_m, COALESCED_DIMENSION(this->number_of_points));
    transpose<<<transpose_grid, transpose_threads>>> (function_values_transposed_local.data,
        function_values.data, COALESCED_DIMENSION(this->number_of_points), group_m);
    function_values_transposed_ptr = function_values_transposed_local.data;
    if (fortran_vars.do_forces || fortran_vars.gga) {
      gradient_values_transposed_local.resize( group_m,COALESCED_DIMENSION(this->number_of_points));
      transpose<<<transpose_grid, transpose_threads>>> (gradient_values_transposed_local.data,
          gradient_values.data, COALESCED_DIMENSION(this->number_of_points), group_m );
      gradient_values_transposed_ptr = gradient_values_transposed_local.data;
    }
  }
  
  HostMatrix<scalar_type> rmm_input_cpu(COALESCED_DIMENSION(group_m), group_m+DENSITY_BLOCK_SIZE);
  get_rmm_input(rmm_input_cpu); //Achica la matriz densidad a la version reducida del grupo

  for (uint i=0; i<(group_m+DENSITY_BLOCK_SIZE); i++)
  {
    for(uint j=0; j<COALESCED_DIMENSION(group_m); j++)
    {
      if((i>=group_m) || (j>=group_m) || (j > i))
      {
        rmm_input_cpu.data[COALESCED_DIMENSION(group_m)*i+j]=0.0f;
      }
    }
  }

  /*
   **********************************************************************
   * Pasando RDM (rmm) a texturas
   **********************************************************************
   */

  // Reuse the per-group cuArray + cudaTextureObject across SCF iterations.
  // Allocation depends only on group_m, which is constant for the lifetime of
  // the group. Only the data changes per iter (re-uploaded below).
  cudaArray* cuArray = reinterpret_cast<cudaArray*>(this->cached_cuArray);
  if (cuArray == NULL || this->cached_rmm_w != rmm_input_cpu.width ||
      this->cached_rmm_h != rmm_input_cpu.height) {
    if (cuArray != NULL) {
      if (this->cached_tex) {
        cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(this->cached_tex));
        this->cached_tex = 0;
      }
      cudaFreeArray(cuArray);
      cuArray = NULL;
      this->cached_cuArray = NULL;
    }
    cudaChannelFormatDesc channelDesc;
#if FULL_DOUBLE
    channelDesc = cudaCreateChannelDesc<int2>();
#else
    channelDesc = cudaCreateChannelDesc<float>();
#endif
    cudaMallocArray(&cuArray, &channelDesc, rmm_input_cpu.width, rmm_input_cpu.height);
    this->cached_cuArray = cuArray;
    this->cached_rmm_w = rmm_input_cpu.width;
    this->cached_rmm_h = rmm_input_cpu.height;

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t new_tex = 0;
    cudaCreateTextureObject(&new_tex, &resDesc, &texDesc, NULL);
    this->cached_tex = static_cast<unsigned long long>(new_tex);
  }
  cudaMemcpyToArray(cuArray, 0, 0, rmm_input_cpu.data,
                    sizeof(scalar_type) * rmm_input_cpu.width * rmm_input_cpu.height,
                    cudaMemcpyHostToDevice);
  cudaTextureObject_t rmm_input_gpu_tex = static_cast<cudaTextureObject_t>(this->cached_tex);

#if USE_LIBXC
  fortran_vars.fexc = fortran_vars.func_coef[0];
#define libxc_init_param \
  fortran_vars.func_id, fortran_vars.func_coef, fortran_vars.nx_func, \
  fortran_vars.nc_func, fortran_vars.nsr_id, fortran_vars.screen, \
  XC_UNPOLARIZED
  LibxcProxy_cuda<scalar_type,4> libxcProxy_cuda(libxc_init_param);
#undef libxc_init_param
#endif

  // For CDFT and becke partitioning.
  CudaMatrix<scalar_type> becke_w_gpu;
  CudaMatrix<scalar_type> cdft_factors_gpu;
  if (((my_cdft_vars.do_chrg || my_cdft_vars.do_spin) && compute_rmm) ||
       (fortran_vars.becke && compute_energy)) {
    becke_w_gpu.resize(fortran_vars.atoms * this->number_of_points);
    HostMatrix<scalar_type> becke_w_cpu(fortran_vars.atoms * this->number_of_points);

    for (unsigned int jpoint = 0; jpoint < this->number_of_points; jpoint++) {
      for (unsigned int iatom = 0; iatom < fortran_vars.atoms; iatom++) {
        becke_w_cpu(jpoint * fortran_vars.atoms + iatom) =
                          (scalar_type) this->points[jpoint].atom_weights(iatom);
     }
    }
    becke_w_gpu = becke_w_cpu;
  }

  if (compute_energy) {
    CudaMatrix<scalar_type> energy_gpu(this->number_of_points);

#define compute_parameters \
    rmm_input_gpu_tex, this->number_of_points, function_values_transposed_ptr, \
    gradient_values_transposed_ptr, hessian_values_transposed.data, group_m, partial_densities_gpu.data, dxyz_gpu.data, \
    dd1_gpu.data,dd2_gpu.data

#define accumulate_parameters \
    energy_gpu.data, factors_gpu.data, point_weights_gpu.data, this->number_of_points, block_height, \
    partial_densities_gpu.data, dxyz_gpu.data, dd1_gpu.data, dd2_gpu.data, fortran_vars.fexc

    if (compute_forces || compute_rmm) {
      if (lda) {
        gpu_compute_density<scalar_type, true><<<threadGrid, threadBlock>>>(compute_parameters);
        gpu_accumulate_point<scalar_type, true, true, true><<<threadGrid_accumulate, threadBlock_accumulate>>> (accumulate_parameters);
      } else {
        gpu_compute_density<scalar_type, false><<<threadGrid, threadBlock>>>(compute_parameters);
#if USE_LIBXC
	      if (fortran_vars.use_libxc) {
	        // Accumulate the data for libxc
	        gpu_accumulate_point_for_libxc<scalar_type, true, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (
		          point_weights_gpu.data, this->number_of_points, block_height,
		          partial_densities_gpu.data, dxyz_gpu.data, dd1_gpu.data, dd2_gpu.data,
		          accumulated_densities_gpu.data, dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);

	        // Compute exc_corr and y2a with libxc GPU version.
	        libxc_exchange_correlation_gpu<scalar_type, true, true, false> (&libxcProxy_cuda,
	          	energy_gpu.data, factors_gpu.data, this->number_of_points,
	        	  accumulated_densities_gpu.data, dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);

	        // Merge the results.
	        gpu_accumulate_energy_and_forces_from_libxc<scalar_type, true, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (
	          	energy_gpu.data, factors_gpu.data, point_weights_gpu.data, this->number_of_points, accumulated_densities_gpu.data);
	      } else {
          gpu_accumulate_point<scalar_type, true, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (accumulate_parameters);
	      }
#else
	      gpu_accumulate_point<scalar_type, true, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (accumulate_parameters);
#endif
      }

      // Constrained DFT accumulation.
      if (my_cdft_vars.do_chrg) {
        cdft_factors_gpu.resize(my_cdft_vars.regions * this->number_of_points);
        cdft_factors_gpu.zero();

        CudaMatrix<uint> cdft_atoms(my_cdft_vars.atoms);
        CudaMatrix<uint> cdft_natom(my_cdft_vars.natom);

        gpu_cdft_factors<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                            cdft_factors_gpu.data, cdft_natom.data, 
                                            cdft_atoms.data,  point_weights_gpu.data,
                                            becke_w_gpu.data, this->number_of_points,
                                            fortran_vars.atoms, my_cdft_vars.regions, my_cdft_vars.max_nat);
      }
    } else {
      if (lda) {
        gpu_compute_density<scalar_type, true><<<threadGrid, threadBlock>>>(compute_parameters);
        gpu_accumulate_point<scalar_type, true, false, true><<<threadGrid_accumulate, threadBlock_accumulate>>> (accumulate_parameters);
      } else {
        gpu_compute_density<scalar_type, false><<<threadGrid, threadBlock>>>(compute_parameters);
#if USE_LIBXC
        if (fortran_vars.use_libxc) {
      	  // Accumulate the data.
	        gpu_accumulate_point_for_libxc<scalar_type, true, false, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (
              point_weights_gpu.data, this->number_of_points, block_height, partial_densities_gpu.data,
              dxyz_gpu.data, dd1_gpu.data, dd2_gpu.data, accumulated_densities_gpu.data,
              dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);
              
	        // Compute exc_corr and y2a with libxc GPU version.
	        libxc_exchange_correlation_gpu<scalar_type, true, true, false> (&libxcProxy_cuda,
            energy_gpu.data, factors_gpu.data, this->number_of_points, accumulated_densities_gpu.data,
            dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);
	        // Merge the results.
	        gpu_accumulate_energy_and_forces_from_libxc<scalar_type, true, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (
	          energy_gpu.data, factors_gpu.data, point_weights_gpu.data, this->number_of_points, accumulated_densities_gpu.data);
	      } else {
          gpu_accumulate_point<scalar_type, true, false, false><<<threadGrid_accumulate, threadBlock_accumulate>>>(accumulate_parameters);
        }
#else
        gpu_accumulate_point<scalar_type, true, false, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (accumulate_parameters);
#endif
      }
    }
    cudaAssertNoError("compute_density");

    HostMatrix<scalar_type> energy_cpu(energy_gpu);
    for (uint i = 0; i < this->number_of_points; i++) {
      energy += energy_cpu(i);
    }

    // Becke partitioning.
    if (fortran_vars.becke) {
      CudaMatrix<scalar_type> becke_dens_gpu(fortran_vars.atoms * this->number_of_points);
      becke_dens_gpu.zero();
      gpu_compute_becke_cs<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(becke_dens_gpu.data,
                                          partial_densities_gpu.data, point_weights_gpu.data, becke_w_gpu.data,
                                          this->number_of_points, fortran_vars.atoms, block_height);

      HostMatrix<scalar_type> becke_dens_cpu(becke_dens_gpu);
      for (unsigned int jpoint = 0; jpoint < this->number_of_points; jpoint++) {
        for (unsigned int iatom = 0; iatom < fortran_vars.atoms; iatom++) {
          becke_dens(iatom) += (double) becke_dens_cpu(jpoint * fortran_vars.atoms + iatom);
        }
      }
    }
  } else {
#undef compute_parameters
#undef accumulate_parameters

#define compute_parameters \
    rmm_input_gpu_tex,this->number_of_points,function_values_transposed_ptr,gradient_values_transposed_ptr,hessian_values_transposed.data,group_m,partial_densities_gpu.data,dxyz_gpu.data,dd1_gpu.data,dd2_gpu.data
#define accumulate_parameters \
    NULL,factors_gpu.data,point_weights_gpu.data,this->number_of_points,block_height,partial_densities_gpu.data,dxyz_gpu.data,dd1_gpu.data,dd2_gpu.data, fortran_vars.fexc
    if (lda)
    {
        gpu_compute_density<scalar_type, true><<<threadGrid, threadBlock>>>(compute_parameters);
        gpu_accumulate_point<scalar_type, false, true, true><<<threadGrid_accumulate, threadBlock_accumulate>>>(accumulate_parameters);
    }
    else
    {
        gpu_compute_density<scalar_type, false><<<threadGrid, threadBlock>>>(compute_parameters);
#if USE_LIBXC
        if (fortran_vars.use_libxc) {
	  // Accumulate the data.
	  gpu_accumulate_point_for_libxc<scalar_type, false, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (point_weights_gpu.data,
            this->number_of_points, block_height,
	    partial_densities_gpu.data, dxyz_gpu.data, dd1_gpu.data, dd2_gpu.data,
	    accumulated_densities_gpu.data, dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);

	  // Compute exc_corr and y2a with libxc GPU version.
	  libxc_exchange_correlation_gpu<scalar_type, false, true, false> (&libxcProxy_cuda,
	    NULL, factors_gpu.data, this->number_of_points,
	    accumulated_densities_gpu.data, dxyz_accum_gpu.data, dd1_accum_gpu.data, dd2_accum_gpu.data);

	  // Merge the results.
	  gpu_accumulate_energy_and_forces_from_libxc<scalar_type, false, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (
	    NULL,factors_gpu.data, point_weights_gpu.data, this->number_of_points, accumulated_densities_gpu.data);
	} else {
    	  gpu_accumulate_point<scalar_type, false, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>>(accumulate_parameters);
	}
#else
        gpu_accumulate_point<scalar_type, false, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>>(accumulate_parameters);
#endif
    }
    if (my_cdft_vars.do_chrg) {
      cdft_factors_gpu.resize(my_cdft_vars.regions * this->number_of_points);
      cdft_factors_gpu.zero();

      CudaMatrix<uint> cdft_atoms(my_cdft_vars.atoms);
      CudaMatrix<uint> cdft_natom(my_cdft_vars.natom);    

      gpu_cdft_factors<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                          cdft_factors_gpu.data, cdft_natom.data, 
                                          cdft_atoms.data,  point_weights_gpu.data,
                                          becke_w_gpu.data, this->number_of_points,
                                          fortran_vars.atoms, my_cdft_vars.regions, my_cdft_vars.max_nat);
    }    
    cudaAssertNoError("compute_density");
  }
#undef compute_parameters
#undef accumulate_parameters

  timers.density.pause_and_sync();
  /* compute forces */
  if (compute_forces) {
    //************ Repongo los valores que puse a cero antes, para las fuerzas son necesarios (o por lo mens utiles)
    for (uint i=0; i<(group_m); i++) {
      for(uint j=0; j<(group_m); j++) {
        if((i>=group_m) || (j>=group_m) || (j > i))
        {
          rmm_input_cpu.data[COALESCED_DIMENSION(group_m)*i+j]=rmm_input_cpu.data[COALESCED_DIMENSION(group_m)*j+i] ;
        }
      }
    }

    timers.density_derivs.start_and_sync();
    cudaMemcpyToArray(cuArray, 0, 0,rmm_input_cpu.data,
      sizeof(scalar_type)*rmm_input_cpu.width*rmm_input_cpu.height, cudaMemcpyHostToDevice);

    timers.density_derivs.start_and_sync();
    dim3 threads = dim3(this->number_of_points);
    threadBlock = dim3(DENSITY_DERIV_BLOCK_SIZE);
    threadGrid = divUp(threads, threadBlock);

    CudaMatrix<vec_type4> dd_gpu(COALESCED_DIMENSION(this->number_of_points), this->total_nucleii()); dd_gpu.zero();
    CudaMatrixUInt nuc_gpu(this->func2local_nuc);  // TODO: esto en realidad se podria guardar una sola vez durante su construccion

    gpu_compute_density_derivs<<<threadGrid, threadBlock>>>(
        rmm_input_gpu_tex, function_values.data, gradient_values.data, nuc_gpu.data, dd_gpu.data, this->number_of_points, group_m, this->total_nucleii());
    cudaAssertNoError("density_derivs");
    timers.density_derivs.pause_and_sync();

    timers.forces.start_and_sync();
    CudaMatrix<vec_type4> forces_gpu(this->total_nucleii());
    forces_gpu.zero();

    threads = dim3(this->total_nucleii());
    threadBlock = dim3(FORCE_BLOCK_SIZE);
    threadGrid = divUp(threads, threadBlock);
    gpu_compute_forces<<<threadGrid, threadBlock>>>(
        this->number_of_points, factors_gpu.data, dd_gpu.data, forces_gpu.data, this->total_nucleii());
    cudaAssertNoError("forces");

    HostMatrix<vec_type4> forces_cpu(forces_gpu);

    for (uint i = 0; i < this->total_nucleii(); ++i) {
      vec_type4 atom_force = forces_cpu(i);
      uint global_nuc = this->local2global_nuc[i];
      fort_forces_ms(global_nuc, 0) += atom_force.x;
      fort_forces_ms(global_nuc, 1) += atom_force.y;
      fort_forces_ms(global_nuc, 2) += atom_force.z;

    }
    timers.forces.pause_and_sync();
  }

  timers.rmm.start_and_sync();
  /* compute RMM */
  if (compute_rmm) {
    threadBlock = dim3(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
    uint blocksPerRow = divUp(group_m, RMM_BLOCK_SIZE_XY);
    // Only use enough blocks for lower triangle
    threadGrid = dim3(blocksPerRow*(blocksPerRow+1)/2);

    CudaMatrix<scalar_type> rmm_output_gpu(COALESCED_DIMENSION(group_m), group_m);
    rmm_output_gpu.zero();


    // Adds CDFT terms to RMM factors.
    CudaMatrix<scalar_type> cdft_Vc;
    if (my_cdft_vars.do_chrg) {
      HostMatrix<scalar_type> cdft_Vc_cpu(my_cdft_vars.regions);

      cdft_Vc.resize(my_cdft_vars.regions);
      for (unsigned int i = 0; i < my_cdft_vars.regions; i++) {
        cdft_Vc_cpu(i) = (scalar_type) my_cdft_vars.Vc(i);
      }
      cdft_Vc = cdft_Vc_cpu;
   
      gpu_cdft_factors_accum<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                                  cdft_factors_gpu.data, this->number_of_points,
                                                  my_cdft_vars.regions, cdft_Vc.data, factors_gpu.data);
    }


    // For calls with a single block (pretty common with cubes) don't bother doing the arithmetic to get block position in the matrix
    if (blocksPerRow > 1) {
        gpu_update_rmm<scalar_type,true><<<threadGrid, threadBlock>>>(factors_gpu.data, this->number_of_points,
                                                                      rmm_output_gpu.data, function_values.data,
                                                                      group_m);
    } else {
        gpu_update_rmm<scalar_type,false><<<threadGrid, threadBlock>>>(factors_gpu.data, this->number_of_points,
                                                                       rmm_output_gpu.data, function_values.data,
                                                                       group_m);
    }

    cudaAssertNoError("update_rmm");

    /*** Scatter local Fock to global packed Fock on GPU ***/
    {
      const unsigned int n_indexes = this->rmm_bigs.size();
      if (n_indexes > 0) {
        if (!this->rmm_bigs_gpu.is_allocated()) {
          this->rmm_bigs_gpu.resize(n_indexes, 1);
          this->rmm_rows_gpu.resize(n_indexes, 1);
          this->rmm_cols_gpu.resize(n_indexes, 1);
          cudaMemcpy(this->rmm_bigs_gpu.data, this->rmm_bigs.data(),
                     n_indexes * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(this->rmm_rows_gpu.data, this->rmm_rows.data(),
                     n_indexes * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(this->rmm_cols_gpu.data, this->rmm_cols.data(),
                     n_indexes * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        }
        const unsigned int rmm_width = COALESCED_DIMENSION(group_m);
        dim3 scatter_block(256);
        dim3 scatter_grid((n_indexes + 255) / 256);
        gpu_scatter_rmm<scalar_type><<<scatter_grid, scatter_block>>>(
            rmm_output_gpu.data, this->rmm_bigs_gpu.data,
            this->rmm_rows_gpu.data, this->rmm_cols_gpu.data,
            s_global_fock_dev.data, n_indexes, rmm_width);
        cudaAssertNoError("gpu_scatter_rmm");
      }
    }
  }
  timers.rmm.pause_and_sync();

  /* clear functions */
  if(!(this->inGlobal)) {
    function_values.deallocate();
    gradient_values.deallocate();
    hessian_values_transposed.deallocate();
  }
  // cuArray + textureObject are now cached on the group; freed in dtor.
}

//======================
// OPENSHELL
//======================

template<class scalar_type>
void PointGroupGPU<scalar_type>::solve_opened(
    Timers& timers, bool compute_rmm, bool lda, bool compute_forces,
    bool compute_energy, double& energy, double& energy_i,
    double& energy_c, double& energy_c1, double& energy_c2,
    HostMatrix<double>& fort_forces_ms, HostMatrix<double>& rmm_output_local_a,
    HostMatrix<double>& rmm_output_local_b, HostMatrix<double>& becke_dens,
    HostMatrix<double>& becke_spin, CDFTVars& my_cdft_vars){

  int device;
  cudaGetDevice(&device);
  current_device = device;

  /*** Computo sobre cada cubo ****/
  /** Compute this group's functions **/
  timers.functions.start_and_sync();
  compute_functions(compute_forces, !lda);
  timers.functions.pause_and_sync();

  uint group_m = this->total_functions();

  timers.density.start_and_sync();
  /** Load points from group (one-time upload — points[].weight is constant
   ** across all SCF iterations). **/
  if (!this->point_weights_gpu_cached.is_allocated() ||
      this->point_weights_gpu_cached.width != this->number_of_points) {
    HostMatrix<scalar_type> point_weights_cpu(this->number_of_points, 1);
    for (uint pi = 0; pi < this->number_of_points; ++pi) {
      point_weights_cpu(pi) = this->points[pi].weight;
    }
    this->point_weights_gpu_cached = point_weights_cpu;
  }
  CudaMatrix<scalar_type>& point_weights_gpu = this->point_weights_gpu_cached;

  dim3 threadBlock, threadGrid;
  const int block_height= divUp(group_m,2*DENSITY_BLOCK_SIZE);

  // This makes sure the amount of functions fits within range.
  threadBlock = dim3(DENSITY_BLOCK_SIZE,1,1);
  threadGrid = dim3(this->number_of_points,block_height,1);

  CudaMatrix<scalar_type> factors_a_gpu;
  CudaMatrix<scalar_type> factors_b_gpu;

  // Gradients (dxyz) and Hessians (dd1,dd2) for alpha/beta.
  CudaMatrix<scalar_type> partial_densities_a_gpu;
  CudaMatrix<vec_type<scalar_type,4> > dxyz_a_gpu;
  CudaMatrix<vec_type<scalar_type,4> > dd1_a_gpu;
  CudaMatrix<vec_type<scalar_type,4> > dd2_a_gpu;

  CudaMatrix<scalar_type> partial_densities_b_gpu;
  CudaMatrix<vec_type<scalar_type,4> > dxyz_b_gpu;
  CudaMatrix<vec_type<scalar_type,4> > dd1_b_gpu;
  CudaMatrix<vec_type<scalar_type,4> > dd2_b_gpu;

  // Matrix transpose is needed for better coalescence in density. Reuse the
  // per-group cached transposed buffers populated once in compute_functions()
  // when inGlobal. Otherwise allocate locally and transpose each iteration.
  CudaMatrix<scalar_type> function_values_transposed_local;
  CudaMatrix<vec_type<scalar_type,4> > gradient_values_transposed_local;
  scalar_type*   function_values_transposed_ptr;
  vec_type<scalar_type,4>* gradient_values_transposed_ptr = NULL;

  int transposed_width = COALESCED_DIMENSION(this->number_of_points);
  #ifndef BLOCK_DIM
  #define BLOCK_DIM 16
  #endif
  dim3 transpose_grid(transposed_width / BLOCK_DIM, divUp((group_m),BLOCK_DIM));
  dim3 transpose_threads(BLOCK_DIM, BLOCK_DIM, 1);

  if (this->inGlobal && this->function_values_transposed_cached.is_allocated()) {
    function_values_transposed_ptr = this->function_values_transposed_cached.data;
    if (fortran_vars.do_forces || fortran_vars.gga) {
      gradient_values_transposed_ptr = this->gradient_values_transposed_cached.data;
    }
  } else {
    function_values_transposed_local.resize(group_m, COALESCED_DIMENSION(this->number_of_points));
    transpose<<<transpose_grid, transpose_threads>>> (function_values_transposed_local.data, function_values.data, COALESCED_DIMENSION(this->number_of_points), group_m);
    function_values_transposed_ptr = function_values_transposed_local.data;
    if (fortran_vars.do_forces || fortran_vars.gga) {
      gradient_values_transposed_local.resize(group_m, COALESCED_DIMENSION(this->number_of_points));
      transpose<<<transpose_grid, transpose_threads>>> (gradient_values_transposed_local.data, gradient_values.data, COALESCED_DIMENSION(this->number_of_points), group_m);
      gradient_values_transposed_ptr = gradient_values_transposed_local.data;
    }
  }

  partial_densities_a_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dxyz_a_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height);
  dd1_a_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height );
  dd2_a_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height );

  partial_densities_b_gpu.resize(COALESCED_DIMENSION(this->number_of_points), block_height);
  dxyz_b_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height);
  dd1_b_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height );
  dd2_b_gpu.resize(COALESCED_DIMENSION(this->number_of_points),block_height );

  const dim3 threadGrid_accumulate(divUp(this->number_of_points,DENSITY_ACCUM_BLOCK_SIZE),1,1);
  const dim3 threadBlock_accumulate(DENSITY_ACCUM_BLOCK_SIZE,1,1);

  if (compute_rmm || compute_forces) {
    factors_a_gpu.resize(this->number_of_points);
    factors_b_gpu.resize(this->number_of_points);
    factors_a_gpu.zero();
    factors_b_gpu.zero();
  }

  HostMatrix<scalar_type> rmm_input_a_cpu(COALESCED_DIMENSION(group_m), group_m+DENSITY_BLOCK_SIZE);
  HostMatrix<scalar_type> rmm_input_b_cpu(COALESCED_DIMENSION(group_m), group_m+DENSITY_BLOCK_SIZE);
   //Reduces density matrixes (Up,Down) to the reduced group version
  get_rmm_input(rmm_input_a_cpu, rmm_input_b_cpu);

  for (uint i=0; i<(group_m+DENSITY_BLOCK_SIZE); i++) {
    for(uint j=0; j<COALESCED_DIMENSION(group_m); j++) {
      if((i>=group_m) || (j>=group_m) || (j > i)) {
        rmm_input_a_cpu.data[COALESCED_DIMENSION(group_m)*i+j]=0.0f;
        rmm_input_b_cpu.data[COALESCED_DIMENSION(group_m)*i+j]=0.0f;
      }
    }
  }

  /*
  **********************************************************************
  * Pasando RDM (rmm) a texturas/
  **********************************************************************
  */

  // Reuse per-group cuArrays + cudaTextureObjects across SCF iterations.
  cudaArray* cuArray1 = reinterpret_cast<cudaArray*>(this->cached_cuArray);
  cudaArray* cuArray2 = reinterpret_cast<cudaArray*>(this->cached_cuArray_b);
  if (cuArray1 == nullptr || this->cached_rmm_w != rmm_input_a_cpu.width ||
      this->cached_rmm_h != rmm_input_a_cpu.height) {
    if (this->cached_tex) {
      cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(this->cached_tex));
      this->cached_tex = 0;
    }
    if (this->cached_tex_b) {
      cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(this->cached_tex_b));
      this->cached_tex_b = 0;
    }
    if (cuArray1) cudaFreeArray(cuArray1);
    if (cuArray2) cudaFreeArray(cuArray2);
    cuArray1 = nullptr; 
    cuArray2 = nullptr;

    cudaChannelFormatDesc channelDesc;
#if FULL_DOUBLE
    channelDesc = cudaCreateChannelDesc<int2>();
#else
    channelDesc = cudaCreateChannelDesc<float>();
#endif
    cudaMallocArray(&cuArray1, &channelDesc, rmm_input_a_cpu.width, rmm_input_a_cpu.height);
    cudaMallocArray(&cuArray2, &channelDesc, rmm_input_b_cpu.width, rmm_input_b_cpu.height);
    this->cached_cuArray   = cuArray1;
    this->cached_cuArray_b = cuArray2;
    this->cached_rmm_w = rmm_input_a_cpu.width;
    this->cached_rmm_h = rmm_input_a_cpu.height;

    cudaResourceDesc resDesc1;
    memset(&resDesc1, 0, sizeof(resDesc1));
    resDesc1.resType = cudaResourceTypeArray;
    resDesc1.res.array.array = cuArray1;
    cudaResourceDesc resDesc2;
    memset(&resDesc2, 0, sizeof(resDesc2));
    resDesc2.resType = cudaResourceTypeArray;
    resDesc2.res.array.array = cuArray2;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t rmm_input_gpu_tex = 0, rmm_input_gpu_tex2 = 0;
    cudaCreateTextureObject(&rmm_input_gpu_tex, &resDesc1, &texDesc, NULL);
    cudaCreateTextureObject(&rmm_input_gpu_tex2, &resDesc2, &texDesc, NULL);
    this->cached_tex   = static_cast<unsigned long long>(rmm_input_gpu_tex);
    this->cached_tex_b = static_cast<unsigned long long>(rmm_input_gpu_tex2);
  }
  cudaMemcpyToArray(cuArray1, 0, 0, rmm_input_a_cpu.data,
                    sizeof(scalar_type) * rmm_input_a_cpu.width * rmm_input_a_cpu.height,
                    cudaMemcpyHostToDevice);
  cudaMemcpyToArray(cuArray2, 0, 0, rmm_input_b_cpu.data,
                    sizeof(scalar_type) * rmm_input_b_cpu.width * rmm_input_b_cpu.height,
                    cudaMemcpyHostToDevice);
  cudaTextureObject_t rmm_input_gpu_tex  = static_cast<cudaTextureObject_t>(this->cached_tex);
  cudaTextureObject_t rmm_input_gpu_tex2 = static_cast<cudaTextureObject_t>(this->cached_tex_b);

  // For CDFT and becke partitioning.
  CudaMatrix<scalar_type> becke_w_gpu;
  CudaMatrix<scalar_type> cdft_factors_gpu;
  if (((my_cdft_vars.do_chrg || my_cdft_vars.do_spin) && compute_rmm) ||
      (fortran_vars.becke && compute_energy)) {
    becke_w_gpu.resize(fortran_vars.atoms * this->number_of_points);
    HostMatrix<scalar_type> becke_w_cpu(fortran_vars.atoms * this->number_of_points);

    for (unsigned int jpoint = 0; jpoint < this->number_of_points; jpoint++) {
      for (unsigned int iatom = 0; iatom < fortran_vars.atoms; iatom++) {
        becke_w_cpu(jpoint * fortran_vars.atoms + iatom) =
                          (scalar_type) this->points[jpoint].atom_weights(iatom);
    }
    }
    becke_w_gpu = becke_w_cpu;
  }

  if (compute_energy) {
    CudaMatrix<scalar_type> energy_gpu(this->number_of_points);

    if (compute_forces || compute_rmm) {
      gpu_compute_density_opened<scalar_type, false><<<threadGrid, threadBlock>>>(
             rmm_input_gpu_tex, rmm_input_gpu_tex2,
             this->number_of_points, function_values_transposed_ptr,
             gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
             partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
             partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
      gpu_accumulate_point_open<scalar_type, true, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (
             energy_gpu.data,
             factors_a_gpu.data, factors_b_gpu.data, point_weights_gpu.data,this->number_of_points,block_height,
             partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
             partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data, fortran_vars.fexc);
      if (my_cdft_vars.do_chrg || my_cdft_vars.do_spin ) {        
        cdft_factors_gpu.resize(my_cdft_vars.regions * this->number_of_points);
        cdft_factors_gpu.zero();

        CudaMatrix<uint> cdft_atoms(my_cdft_vars.atoms);
        CudaMatrix<uint> cdft_natom(my_cdft_vars.natom);    

        gpu_cdft_factors<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                            cdft_factors_gpu.data, cdft_natom.data, 
                                            cdft_atoms.data,  point_weights_gpu.data,
                                            becke_w_gpu.data, this->number_of_points,
                                            fortran_vars.atoms, my_cdft_vars.regions, my_cdft_vars.max_nat);
      }
    } else {
      gpu_compute_density_opened<scalar_type, false><<<threadGrid, threadBlock>>>(
             rmm_input_gpu_tex, rmm_input_gpu_tex2,
             this->number_of_points, function_values_transposed_ptr,
             gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
             partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
             partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
      gpu_accumulate_point_open<scalar_type, true, false, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (
             energy_gpu.data, factors_a_gpu.data, factors_b_gpu.data, point_weights_gpu.data,
             this->number_of_points, block_height,
             partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
             partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data, fortran_vars.fexc);
    }
    cudaAssertNoError("compute_density");

    HostMatrix<scalar_type> energy_cpu(energy_gpu);

    for (uint i = 0; i < this->number_of_points; i++) {
      energy    += energy_cpu(i);
    }

     // Becke partitioning.
     if (fortran_vars.becke) {   
      CudaMatrix<scalar_type> becke_dens_gpu(fortran_vars.atoms * this->number_of_points);
      CudaMatrix<scalar_type> becke_spin_gpu(fortran_vars.atoms * this->number_of_points);
      becke_dens_gpu.zero();
      becke_spin_gpu.zero();
      gpu_compute_becke_os<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>
                          (becke_dens_gpu.data, becke_spin_gpu.data, partial_densities_a_gpu.data,
                           partial_densities_b_gpu.data, point_weights_gpu.data, becke_w_gpu.data,
                           this->number_of_points, fortran_vars.atoms, block_height);

      HostMatrix<scalar_type> becke_dens_cpu(becke_dens_gpu);
      HostMatrix<scalar_type> becke_spin_cpu(becke_spin_gpu);
      for (unsigned int jpoint = 0; jpoint < this->number_of_points; jpoint++) {
        for (unsigned int iatom = 0; iatom < fortran_vars.atoms; iatom++) {
          becke_dens(iatom) += (double) becke_dens_cpu(jpoint * fortran_vars.atoms + iatom);
          becke_spin(iatom) += (double) becke_spin_cpu(jpoint * fortran_vars.atoms + iatom);
        }
      }
    }
  } else {
    gpu_compute_density_opened<scalar_type, false><<<threadGrid, threadBlock>>>(
           rmm_input_gpu_tex, rmm_input_gpu_tex2,
           this->number_of_points, function_values_transposed_ptr,
           gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
           partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
           partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
    gpu_accumulate_point_open<scalar_type, false, true, false><<<threadGrid_accumulate, threadBlock_accumulate>>> (
           NULL,
           factors_a_gpu.data, factors_b_gpu.data, point_weights_gpu.data,this->number_of_points,block_height,
           partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
           partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data, fortran_vars.fexc);

    if (my_cdft_vars.do_chrg || my_cdft_vars.do_spin ) {           
      cdft_factors_gpu.resize(my_cdft_vars.regions * this->number_of_points);
      cdft_factors_gpu.zero();

      CudaMatrix<uint> cdft_atoms(my_cdft_vars.atoms);
      CudaMatrix<uint> cdft_natom(my_cdft_vars.natom);    

      gpu_cdft_factors<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                          cdft_factors_gpu.data, cdft_natom.data, 
                                          cdft_atoms.data,  point_weights_gpu.data,
                                          becke_w_gpu.data, this->number_of_points,
                                          fortran_vars.atoms, my_cdft_vars.regions, my_cdft_vars.max_nat);
    }
    cudaAssertNoError("compute_density");
  }

  timers.density.pause_and_sync();


  /* compute forces */
  if (compute_forces) {

    // Repongo los valores que puse a cero antes, para las fuerzas son necesarios (o por lo menos utiles)
    for (uint i=0; i<(group_m); i++) {
    for (uint j=0; j<(group_m); j++) {
      if((i>=group_m) || (j>=group_m) || (j > i)){
        rmm_input_a_cpu.data[COALESCED_DIMENSION(group_m)*i+j] =
                        rmm_input_a_cpu.data[COALESCED_DIMENSION(group_m)*j+i] ;
        rmm_input_b_cpu.data[COALESCED_DIMENSION(group_m)*i+j] =
                        rmm_input_b_cpu.data[COALESCED_DIMENSION(group_m)*j+i] ;
      }
    }
    }

    cudaMemcpyToArray(cuArray1, 0, 0,rmm_input_a_cpu.data,sizeof(scalar_type)*rmm_input_a_cpu.width*rmm_input_a_cpu.height, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuArray2, 0, 0,rmm_input_b_cpu.data,sizeof(scalar_type)*rmm_input_b_cpu.width*rmm_input_b_cpu.height, cudaMemcpyHostToDevice);


    dim3 threads;
    timers.density_derivs.start_and_sync();
    threads = dim3(this->number_of_points);
    threadBlock = dim3(DENSITY_DERIV_BLOCK_SIZE);
    threadGrid = divUp(threads, threadBlock);

    CudaMatrix<vec_type4> dd_gpu_a(COALESCED_DIMENSION(this->number_of_points), this->total_nucleii());
    CudaMatrix<vec_type4> dd_gpu_b(COALESCED_DIMENSION(this->number_of_points), this->total_nucleii());
    dd_gpu_a.zero();
    dd_gpu_b.zero();
    CudaMatrixUInt nuc_gpu(this->func2local_nuc);

    // Kernel
    gpu_compute_density_derivs_open<<<threadGrid, threadBlock>>>(rmm_input_gpu_tex, rmm_input_gpu_tex2, function_values.data, gradient_values.data, nuc_gpu.data, dd_gpu_a.data, dd_gpu_b.data, this->number_of_points, group_m, this->total_nucleii());

    cudaAssertNoError("density_derivs");
    timers.density_derivs.pause_and_sync();

    timers.forces.start_and_sync();
    CudaMatrix<vec_type4> forces_gpu_a(this->total_nucleii());
    CudaMatrix<vec_type4> forces_gpu_b(this->total_nucleii());
    forces_gpu_a.zero();
    forces_gpu_b.zero();

    threads = dim3(this->total_nucleii());
    threadBlock = dim3(FORCE_BLOCK_SIZE);
    threadGrid = divUp(threads, threadBlock);
    // Kernel
    gpu_compute_forces<<<threadGrid, threadBlock>>>(this->number_of_points, factors_a_gpu.data, dd_gpu_a.data, forces_gpu_a.data, this->total_nucleii());
    gpu_compute_forces<<<threadGrid, threadBlock>>>(this->number_of_points, factors_b_gpu.data, dd_gpu_b.data, forces_gpu_b.data, this->total_nucleii());

    cudaAssertNoError("forces");

    HostMatrix<vec_type4> forces_cpu_a(forces_gpu_a);
    HostMatrix<vec_type4> forces_cpu_b(forces_gpu_b);

    for (uint i = 0; i < this->total_nucleii(); ++i) {
      vec_type4 atom_force_a = forces_cpu_a(i);
      vec_type4 atom_force_b = forces_cpu_b(i);
      uint global_nuc = this->local2global_nuc[i];

      fort_forces_ms(global_nuc, 0) += atom_force_a.x + atom_force_b.x;
      fort_forces_ms(global_nuc, 1) += atom_force_a.y + atom_force_b.y;
      fort_forces_ms(global_nuc, 2) += atom_force_a.z + atom_force_b.z;
    }

    timers.forces.pause_and_sync();
  }

  /* compute RMM */
  timers.rmm.start_and_sync();
  if (compute_rmm) {
    threadBlock = dim3(RMM_BLOCK_SIZE_XY, RMM_BLOCK_SIZE_XY);
    uint blocksPerRow = divUp(group_m, RMM_BLOCK_SIZE_XY);

    // Only use enough blocks for lower triangle
    threadGrid = dim3(blocksPerRow*(blocksPerRow+1)/2);
    CudaMatrix<scalar_type> rmm_output_a_gpu(COALESCED_DIMENSION(group_m), group_m);
    CudaMatrix<scalar_type> rmm_output_b_gpu(COALESCED_DIMENSION(group_m), group_m);

    rmm_output_a_gpu.zero();
    rmm_output_b_gpu.zero();

    // Adds CDFT terms to RMM factors.
    CudaMatrix<scalar_type> cdft_Vc;
    CudaMatrix<scalar_type> cdft_Vs_a;
    CudaMatrix<scalar_type> cdft_Vs_b;
    if (my_cdft_vars.do_chrg) {
      HostMatrix<scalar_type> cdft_Vc_cpu(my_cdft_vars.regions);
      cdft_Vc.resize(my_cdft_vars.regions);
      for (unsigned int i = 0; i < my_cdft_vars.regions; i++) {
        cdft_Vc_cpu(i) = (scalar_type) my_cdft_vars.Vc(i);
      }
      cdft_Vc = cdft_Vc_cpu;
      gpu_cdft_factors_accum<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                                  cdft_factors_gpu.data, this->number_of_points,
                                                  my_cdft_vars.regions, cdft_Vc.data, factors_a_gpu.data);
      gpu_cdft_factors_accum<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                                  cdft_factors_gpu.data, this->number_of_points,
                                                  my_cdft_vars.regions, cdft_Vc.data, factors_b_gpu.data);
    }

    if (my_cdft_vars.do_spin) {
      HostMatrix<scalar_type> cdft_Vs_cpu(my_cdft_vars.regions);
      cdft_Vs_a.resize(my_cdft_vars.regions);
      cdft_Vs_b.resize(my_cdft_vars.regions);
      for (unsigned int i = 0; i < my_cdft_vars.regions; i++) {
        cdft_Vs_cpu(i) = (scalar_type) my_cdft_vars.Vs(i);
      }
      cdft_Vs_b = cdft_Vs_cpu;

      for (unsigned int i = 0; i < my_cdft_vars.regions; i++) {
        cdft_Vs_cpu(i) = - (scalar_type) my_cdft_vars.Vs(i);
      }
      cdft_Vs_a = cdft_Vs_cpu;

      gpu_cdft_factors_accum<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                                  cdft_factors_gpu.data, this->number_of_points,
                                                  my_cdft_vars.regions, cdft_Vs_a.data, factors_a_gpu.data);
      gpu_cdft_factors_accum<scalar_type><<<threadGrid_accumulate, threadBlock_accumulate>>>(
                                                  cdft_factors_gpu.data, this->number_of_points,
                                                  my_cdft_vars.regions, cdft_Vs_b.data, factors_b_gpu.data);
    }

    // For calls with a single block (pretty common with cubes) don't bother doing the arithmetic to get block position in the matrix
    if (blocksPerRow > 1) {
      gpu_update_rmm<scalar_type,true><<<threadGrid, threadBlock>>>(factors_a_gpu.data, this->number_of_points,
                                                                    rmm_output_a_gpu.data, function_values.data,
                                                                    group_m);
      gpu_update_rmm<scalar_type,true><<<threadGrid, threadBlock>>>(factors_b_gpu.data, this->number_of_points,
                                                                    rmm_output_b_gpu.data, function_values.data,
                                                                    group_m);
    } else {
      gpu_update_rmm<scalar_type,false><<<threadGrid, threadBlock>>>(factors_a_gpu.data, this->number_of_points,
                                                                     rmm_output_a_gpu.data, function_values.data,
                                                                     group_m);
      gpu_update_rmm<scalar_type,false><<<threadGrid, threadBlock>>>(factors_b_gpu.data, this->number_of_points,
                                                                     rmm_output_b_gpu.data, function_values.data,
                                                                     group_m);
    }

    cudaAssertNoError("update_rmm");

    /*** Scatter local Fock (alpha+beta) to global packed Fock on GPU ***/
    {
      const unsigned int n_indexes = this->rmm_bigs.size();
      if (n_indexes > 0) {
        if (!this->rmm_bigs_gpu.is_allocated()) {
          this->rmm_bigs_gpu.resize(n_indexes, 1);
          this->rmm_rows_gpu.resize(n_indexes, 1);
          this->rmm_cols_gpu.resize(n_indexes, 1);
          cudaMemcpy(this->rmm_bigs_gpu.data, this->rmm_bigs.data(),
                     n_indexes * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(this->rmm_rows_gpu.data, this->rmm_rows.data(),
                     n_indexes * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(this->rmm_cols_gpu.data, this->rmm_cols.data(),
                     n_indexes * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        }
        const unsigned int rmm_width = COALESCED_DIMENSION(group_m);
        dim3 scatter_block(256);
        dim3 scatter_grid((n_indexes + 255) / 256);
        gpu_scatter_rmm<scalar_type><<<scatter_grid, scatter_block>>>(
            rmm_output_a_gpu.data, this->rmm_bigs_gpu.data,
            this->rmm_rows_gpu.data, this->rmm_cols_gpu.data,
            s_global_fock_a_dev.data, n_indexes, rmm_width);
        gpu_scatter_rmm<scalar_type><<<scatter_grid, scatter_block>>>(
            rmm_output_b_gpu.data, this->rmm_bigs_gpu.data,
            this->rmm_rows_gpu.data, this->rmm_cols_gpu.data,
            s_global_fock_b_dev.data, n_indexes, rmm_width);
        cudaAssertNoError("gpu_scatter_rmm_open");
      }
    }
  }
  timers.rmm.pause_and_sync();

  /* clear functions */
  if(!(this->inGlobal)) {
    function_values.deallocate();
    gradient_values.deallocate();
    hessian_values_transposed.deallocate();
  }
  // cuArrays + textureObjects are cached on the group; freed in dtor.

  //uint free_memory, total_memory;
  //cudaGetMemoryInfo(free_memory, total_memory);
  //cout << "Maximum used memory: " << (double)max_used_memory / (1024 * 1024) << "MB (" << ((double)max_used_memory / total_memory) * 100.0 << "%)" << endl;
  //cudaPrintMemoryInfo();
}



/*******************************
 * Cube Functions
 *******************************/

template<class scalar_type>
void PointGroupGPU<scalar_type>::compute_functions(bool forces, bool gga)
{
  if(this->inGlobal) //Ya las tengo en memoria? entonces salgo porque ya estan las 3 calculadas
    return;

  if(0 == GlobalMemoryPool::tryAlloc(this->size_in_gpu())) //1 si hubo error, 0 si pude reservar la memoria
    this->inGlobal=true;
  CudaMatrix<vec_type4> points_position_gpu;
  CudaMatrix<vec_type2> factor_ac_gpu;
  CudaMatrixUInt nuc_gpu;
  CudaMatrixUInt contractions_gpu;

  /** Load points from group **/
  {
    HostMatrix<vec_type4> points_position_cpu(this->number_of_points, 1);
    uint i = 0;
    for (vector<Point>::const_iterator p = this->points.begin(); p != this->points.end(); ++p, ++i) {
      points_position_cpu(i) = vec_type4(p->position.x, p->position.y, p->position.z, 0);
    }
    points_position_gpu = points_position_cpu;
  }
  /* Load group functions */
  uint group_m = this->s_functions + this->p_functions * 3 + this->d_functions * 6;
  uint4 group_functions = make_uint4(this->s_functions, this->p_functions, this->d_functions, group_m);
  HostMatrix<vec_type2> factor_ac_cpu(COALESCED_DIMENSION(group_m), MAX_CONTRACTIONS);
  HostMatrixUInt nuc_cpu(group_m, 1), contractions_cpu(group_m, 1);

  // TODO: hacer que functions.h itere por total_small_functions()... asi puedo hacer que
  // func2global_nuc sea de tamaño total_functions() y directamente copio esa matriz aca y en otros lados

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
        factor_ac_cpu(ii, k) = vec_type2(fortran_vars.a_values(func, k), fortran_vars.c_values(func, k));
      ii++;
    }
  }
  factor_ac_gpu = factor_ac_cpu;
  nuc_gpu = nuc_cpu;
  contractions_gpu = contractions_cpu;

  CudaMatrix<vec_type<scalar_type,4> > hessian_values;
  /** Compute Functions **/
  function_values.resize(COALESCED_DIMENSION(this->number_of_points), group_functions.w);
  function_values.zero();
  if (fortran_vars.do_forces || fortran_vars.gga) {
      gradient_values.resize(COALESCED_DIMENSION(this->number_of_points), group_functions.w);
      gradient_values.zero();
  }
  if (fortran_vars.gga) {
      hessian_values.resize(COALESCED_DIMENSION(this->number_of_points), (group_functions.w) * 2);
      hessian_values.zero();
  }
  
  dim3 threads(this->number_of_points);
  dim3 threadBlock(FUNCTIONS_BLOCK_SIZE);
  dim3 threadGrid = divUp(threads, threadBlock);

#define compute_functions_parameters \
  points_position_gpu.data,this->number_of_points,contractions_gpu.data,factor_ac_gpu.data,nuc_gpu.data,function_values.data,gradient_values.data,hessian_values.data,group_functions
  if (forces) {
    if (gga)
      gpu_compute_functions<scalar_type, true, true><<<threadGrid, threadBlock>>>(compute_functions_parameters);
    else
      gpu_compute_functions<scalar_type, true, false><<<threadGrid, threadBlock>>>(compute_functions_parameters);
  }
  else {
    if (gga)
      gpu_compute_functions<scalar_type, false, true><<<threadGrid, threadBlock>>>(compute_functions_parameters);
    else
      gpu_compute_functions<scalar_type, false, false><<<threadGrid, threadBlock>>>(compute_functions_parameters);
  }

  cudaDeviceSynchronize();

  if (fortran_vars.gga) {
    int transposed_width = COALESCED_DIMENSION(this->number_of_points);
    #define BLOCK_DIM 16
    dim3 transpose_threads(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 transpose_grid=dim3(transposed_width / BLOCK_DIM, divUp((group_m)*2, BLOCK_DIM), 1);
    hessian_values_transposed.resize((group_m) * 2, COALESCED_DIMENSION(this->number_of_points));
    transpose<<<transpose_grid, transpose_threads>>> (hessian_values_transposed.data,
        hessian_values.data, COALESCED_DIMENSION(this->number_of_points), (group_m)*2);
  }

  // Populate cached transposes of function_values / gradient_values when this
  // group's basis caches live across iterations. These outputs feed
  // gpu_compute_density on every subsequent SCF iteration unchanged, so we
  // pay the transpose cost once instead of every iter.
  if (this->inGlobal) {
    int transposed_width = COALESCED_DIMENSION(this->number_of_points);
    #ifndef BLOCK_DIM
    #define BLOCK_DIM 16
    #endif
    dim3 transpose_threads(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 transpose_grid_m(transposed_width / BLOCK_DIM, divUp(group_m, BLOCK_DIM), 1);

    function_values_transposed_cached.resize(group_m, COALESCED_DIMENSION(this->number_of_points));
    transpose<<<transpose_grid_m, transpose_threads>>>(
        function_values_transposed_cached.data, function_values.data,
        COALESCED_DIMENSION(this->number_of_points), group_m);

    if (fortran_vars.do_forces || fortran_vars.gga) {
      gradient_values_transposed_cached.resize(group_m, COALESCED_DIMENSION(this->number_of_points));
      transpose<<<transpose_grid_m, transpose_threads>>>(
          gradient_values_transposed_cached.data, gradient_values.data,
          COALESCED_DIMENSION(this->number_of_points), group_m);
    }
  }

  cudaAssertNoError("compute_functions");
}

/*******************************
 * Cube Weights
 *******************************/
template<class scalar_type>
void PointGroupGPU<scalar_type>::compute_weights(void)
{
  CudaMatrix<vec_type4> point_positions_gpu;
  CudaMatrix<vec_type4> atom_position_rm_gpu;
  {
    HostMatrix<vec_type4> points_positions_cpu(this->number_of_points, 1);
		uint i = 0;
		for (vector<Point>::const_iterator p = this->points.begin(); p != this->points.end(); ++p, ++i) {
			points_positions_cpu(i) = vec_type4(p->position.x, p->position.y, p->position.z, p->atom);
		}
    point_positions_gpu = points_positions_cpu;

    HostMatrix<vec_type4> atom_position_rm_cpu(fortran_vars.atoms, 1);
    for (uint i = 0; i < fortran_vars.atoms; i++) {
      double3 atom_pos = fortran_vars.atom_positions(i);
      atom_position_rm_cpu(i) = vec_type4(atom_pos.x, atom_pos.y, atom_pos.z, fortran_vars.rm(i));
    }
    atom_position_rm_gpu = atom_position_rm_cpu;
  }

  CudaMatrixUInt nucleii_gpu(this->local2global_nuc);

  CudaMatrix<scalar_type> weights_gpu(this->number_of_points);
  dim3 threads(this->number_of_points);
  dim3 blockSize(WEIGHT_BLOCK_SIZE);
  dim3 gridSize = divUp(threads, blockSize);
  gpu_compute_weights<scalar_type><<<gridSize,blockSize>>>(
      this->number_of_points, point_positions_gpu.data, atom_position_rm_gpu.data, weights_gpu.data, nucleii_gpu.data, this->total_nucleii());
  cudaAssertNoError("compute_weights");

  HostMatrix<scalar_type> weights_cpu(weights_gpu);
  uint i = 0;
  for (vector<Point>::iterator p =this->points.begin(); p != this->points.end(); ++p, ++i) {
    p->weight *= weights_cpu(i);
    }
}

#if FULL_DOUBLE
template class PointGroup<double>;
template class PointGroupGPU<double>;
#else
template class PointGroup<float>;
template class PointGroupGPU<float>;
#endif

}
