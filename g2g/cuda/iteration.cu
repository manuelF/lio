/* -*- mode: c -*- */
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <math_constants.h>
#include <mutex>
#include <string>
#include <vector>
#include <cublas_v2.h>

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
#include "kernels/rmm_gather.h"

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

// GPU-side packed-triangular P (density) buffers, mirror of the Fock pattern
// but going the other way: uploaded once per Partition::solve() before the
// parallel region. Each PointGroupGPU::solve_* runs an on-GPU gather kernel
// that reads from these into a per-group local scratch, then a D2D copy
// feeds the density-kernel texture. Replaces the per-group CPU pack +
// Host→Array round-trip that previously fired every SCF/TD iteration.
static CudaMatrix<double> s_global_rdm_dev;
static CudaMatrix<double> s_global_rdm_a_dev;
static CudaMatrix<double> s_global_rdm_b_dev;

// cuBLAS handle used for the GEMM path that replaces the gpu_update_rmm
// kernel on multi-block (larger group_m) groups. The kernel computes
// rmm[i,j] = sum_p factors[p] * f[i,p] * f[j,p] which is A^T * diag(d) * A.
// Implemented as cublas?dgmm (column scale) + cublas?gemm: the bare
// triangular form (?syrkx) picks tiles that fill only 1-2 SMs at our
// problem sizes, so dense GEMM (full m*m output, half of which is unused)
// wins by ~3x on the heuristic search.
static cublasHandle_t s_cublas_handle = nullptr;
static std::once_flag s_cublas_init_flag;
static cublasHandle_t get_cublas_handle() {
  std::call_once(s_cublas_init_flag, []() {
    cublasCreate(&s_cublas_handle);
    cublasSetPointerMode(s_cublas_handle, CUBLAS_POINTER_MODE_HOST);
  });
  return s_cublas_handle;
}
extern "C" void g2g_destroy_cublas() {
  if (s_cublas_handle) {
    cublasDestroy(s_cublas_handle);
    s_cublas_handle = nullptr;
  }
}

// dgmm(LEFT) wrapper: B[r,c] = d[r] * A[r,c]. Column-major view, lda>=m.
static inline cublasStatus_t cublas_dgmm_left(cublasHandle_t h, int m, int n,
                                              const float* A, int lda,
                                              const float* d, float* B, int ldb) {
  return cublasSdgmm(h, CUBLAS_SIDE_LEFT, m, n, A, lda, d, 1, B, ldb);
}
static inline cublasStatus_t cublas_dgmm_left(cublasHandle_t h, int m, int n,
                                              const double* A, int lda,
                                              const double* d, double* B, int ldb) {
  return cublasDdgmm(h, CUBLAS_SIDE_LEFT, m, n, A, lda, d, 1, B, ldb);
}
// gemm wrapper: C = alpha * op(A) * op(B) + beta * C.
static inline cublasStatus_t cublas_gemm(cublasHandle_t h,
                                         cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const float* A, int lda,
                                         const float* B, int ldb,
                                         const float* beta,
                                         float* C, int ldc) {
  return cublasSgemm(h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
static inline cublasStatus_t cublas_gemm(cublasHandle_t h,
                                         cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k,
                                         const double* alpha,
                                         const double* A, int lda,
                                         const double* B, int ldb,
                                         const double* beta,
                                         double* C, int ldc) {
  return cublasDgemm(h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Replacement for the multi-block gpu_update_rmm kernel using cuBLAS.
// function_values is laid out as m rows of COALESCED_DIM(points) cols
// (function-major, point-inner). Viewed in column-major as a (lda x m)
// matrix with lda = COALESCED_DIM(points). With trans=T and k=points,
// cuBLAS only iterates the first `points` entries per column, so padding
// is automatically skipped.
//
// The cached scaled scratch buffer is allocated per group (lazily) and reused
// across iterations — sized lda*m which is the same footprint as
// function_values itself. The CudaMatrix uses the cudaMallocAsync pool so
// growing the high-water-mark only costs in the first few iterations.
template <class scalar_type>
static void gpu_update_rmm_cublas(const scalar_type* factors, int points,
                                  scalar_type* rmm_out, const scalar_type* function_values,
                                  int group_m, int lda, int ldc,
                                  CudaMatrix<scalar_type>& scaled_scratch) {
  cublasHandle_t h = get_cublas_handle();
  // scaled = diag(factors) * function_values (column-major view)
  if (!scaled_scratch.is_allocated() || (int)scaled_scratch.width != lda ||
      (int)scaled_scratch.height < group_m) {
    scaled_scratch.resize(lda, group_m);
  }
  cublas_dgmm_left(h, points, group_m, function_values, lda, factors,
                   scaled_scratch.data, lda);
  const scalar_type alpha = (scalar_type)1.0;
  const scalar_type beta = (scalar_type)0.0;
  // C = function_values^T * scaled (column-major m x m).
  // C[i,j] = sum_p f[p,i] * factor[p]*f[p,j] = sum_p f[i,p]*factor[p]*f[j,p].
  // Scatter only reads the (i <= j) entries — UPPER in column-major — so the
  // duplicate writes to the strictly-lower triangle are harmless.
  cublas_gemm(h, CUBLAS_OP_T, CUBLAS_OP_N, group_m, group_m, points,
              &alpha, function_values, lda,
              scaled_scratch.data, lda, &beta, rmm_out, ldc);
}

// Open-shell fused variant: produces both rmm_a and rmm_b in one larger GEMM.
// Layout: column-major scratch [scaled_a | scaled_b] of size (lda x 2*group_m)
// and column-major output [rmm_a | rmm_b] of size (ldc x 2*group_m). The
// output's first group_m columns hold rmm_a; the next group_m columns hold
// rmm_b. One gemm with n=2*group_m roughly doubles the tile count, which at
// our group sizes (a few SMs' worth of tiles per call) actually helps —
// versus two separate small GEMMs each spawning 1-2 SMs of tiles.
template <class scalar_type>
static void gpu_update_rmm_cublas_open(
    const scalar_type* factors_a, const scalar_type* factors_b, int points,
    scalar_type* rmm_out_combined, const scalar_type* function_values,
    int group_m, int lda, int ldc,
    CudaMatrix<scalar_type>& scaled_scratch_combined) {
  cublasHandle_t h = get_cublas_handle();
  if (!scaled_scratch_combined.is_allocated() ||
      (int)scaled_scratch_combined.width != lda ||
      (int)scaled_scratch_combined.height < 2 * group_m) {
    scaled_scratch_combined.resize(lda, 2 * group_m);
  }
  // First half: scaled_a = diag(factors_a) * function_values
  cublas_dgmm_left(h, points, group_m, function_values, lda, factors_a,
                   scaled_scratch_combined.data, lda);
  // Second half: scaled_b = diag(factors_b) * function_values
  cublas_dgmm_left(h, points, group_m, function_values, lda, factors_b,
                   scaled_scratch_combined.data + (size_t)lda * group_m, lda);
  const scalar_type alpha = (scalar_type)1.0;
  const scalar_type beta = (scalar_type)0.0;
  cublas_gemm(h, CUBLAS_OP_T, CUBLAS_OP_N, group_m, 2 * group_m, points,
              &alpha, function_values, lda,
              scaled_scratch_combined.data, lda, &beta,
              rmm_out_combined, ldc);
}

// Upload the global packed-triangular P matrix (size m*(m+1)/2 doubles) to
// the device once per Partition::solve(). For closed shell pass open=false
// and the *_a/_b pointers as nullptr. For open shell pass open=true and both
// alpha/beta host pointers. Uses cudaMemcpyAsync on stream 0; for closed-shell
// the source is host-pinned via cudaHostRegister in g2g_init_, so this is a
// pure DMA enqueue. Open-shell host buffers are not pinned (registering past
// the Fortran-side allocation broke open-shell SCF — see init.cpp), so the
// driver stages through its pinned buffer once, still a one-shot cost.
void gpu_upload_global_rdm(const double* h_rdm,
                           const double* h_rdm_a,
                           const double* h_rdm_b,
                           unsigned int rmm_global_size,
                           bool open) {
  if (!open) {
    if (!s_global_rdm_dev.is_allocated() ||
        s_global_rdm_dev.width != rmm_global_size) {
      s_global_rdm_dev.resize(rmm_global_size, 1);
    }
    cudaMemcpyAsync(s_global_rdm_dev.data, h_rdm,
                    rmm_global_size * sizeof(double),
                    cudaMemcpyHostToDevice, 0);
  } else {
    if (!s_global_rdm_a_dev.is_allocated() ||
        s_global_rdm_a_dev.width != rmm_global_size) {
      s_global_rdm_a_dev.resize(rmm_global_size, 1);
      s_global_rdm_b_dev.resize(rmm_global_size, 1);
    }
    cudaMemcpyAsync(s_global_rdm_a_dev.data, h_rdm_a,
                    rmm_global_size * sizeof(double),
                    cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(s_global_rdm_b_dev.data, h_rdm_b,
                    rmm_global_size * sizeof(double),
                    cudaMemcpyHostToDevice, 0);
  }
}

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

// Free the per-PointGroupGPU cudaTextureObject cache. Called from
// PointGroupGPU::deallocate() in partition.cpp via an opaque interface so
// partition.cpp does not need to include CUDA runtime headers.
void gpu_release_group_rmm_texture(unsigned long long& tex_handle,
                                   void*& tex_src) {
  if (tex_handle) {
    cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(tex_handle));
    tex_handle = 0;
  }
  tex_src = nullptr;
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

  // Reuse per-group cached scratch. Sizes only depend on number_of_points +
  // block_height which are constant for the group, so resize() is a no-op
  // after the first iteration.
  CudaMatrix<scalar_type>&               partial_densities_gpu = this->partial_densities_a_cached;
  CudaMatrix< vec_type<scalar_type,4> >& dxyz_gpu              = this->dxyz_a_cached;
  CudaMatrix< vec_type<scalar_type,4> >& dd1_gpu               = this->dd1_a_cached;
  CudaMatrix< vec_type<scalar_type,4> >& dd2_gpu               = this->dd2_a_cached;

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

  CudaMatrix<scalar_type>& factors_gpu = this->factors_a_cached;
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
  
  // GPU-side gather of this group's local P from the once-per-solve global
  // packed-triangular buffer (s_global_rdm_dev). Replaces the CPU pack +
  // Host→Array round-trip. Layout: width = COALESCED_DIM(group_m), height =
  // group_m + DENSITY_BLOCK_SIZE; the gather writes only the symmetric
  // (group_m × group_m) block — padding rows/cols past group_m stay zero
  // after the one-shot memset on first allocation. The density kernel's
  // `i < m` + `(bj+j) ≤ i` guards keep it inside that block; the padding is
  // kept zeroed for the forces path (density_derivs), which is less strictly
  // guarded.
  const unsigned int rmm_width = COALESCED_DIMENSION(group_m);
  const unsigned int rmm_height = group_m + DENSITY_BLOCK_SIZE;
  CudaMatrix<scalar_type>& rdm_local_dev = this->rdm_local_dev_a_cached;
  if (!rdm_local_dev.is_allocated() ||
      rdm_local_dev.width != rmm_width || rdm_local_dev.height != rmm_height) {
    rdm_local_dev.resize(rmm_width, rmm_height);
    cudaMemsetAsync(rdm_local_dev.data, 0,
                    sizeof(scalar_type) * rmm_width * rmm_height, 0);
  }
  // Lazy upload of per-group bigs/rows/cols index tables (shared with the
  // scatter side further below).
  const unsigned int n_indexes = this->rmm_bigs.size();
  if (!this->rmm_bigs_gpu.is_allocated() && n_indexes > 0) {
    this->rmm_bigs_gpu.resize(n_indexes, 1);
    this->rmm_rows_gpu.resize(n_indexes, 1);
    this->rmm_cols_gpu.resize(n_indexes, 1);
    cudaMemcpy(this->rmm_bigs_gpu.data, this->rmm_bigs.data(),
               n_indexes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->rmm_rows_gpu.data, this->rmm_rows.data(),
               n_indexes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->rmm_cols_gpu.data, this->rmm_cols.data(),
               n_indexes * sizeof(unsigned int), cudaMemcpyHostToDevice);
  }
  if (n_indexes > 0) {
    dim3 gather_block(256);
    dim3 gather_grid((n_indexes + 255) / 256);
    gpu_gather_rdm<scalar_type><<<gather_grid, gather_block>>>(
        s_global_rdm_dev.data, this->rmm_bigs_gpu.data,
        this->rmm_rows_gpu.data, this->rmm_cols_gpu.data,
        rdm_local_dev.data, n_indexes, rmm_width);
    cudaAssertNoError("gpu_gather_rdm");
  }

  /*
   **********************************************************************
   * Pasando RDM (rmm) a texturas
   **********************************************************************
   */

  // Bind a cudaTextureObject_t directly to the gather output buffer via
  // cudaResourceTypePitch2D. Rebuilt only when the buffer pointer or
  // dimensions change (cudaMallocAsync may return a new ptr on resize).
  void* src = static_cast<void*>(rdm_local_dev.data);
  if (this->cached_tex == 0 ||
      this->cached_tex_src != src ||
      this->cached_rmm_w != rmm_width ||
      this->cached_rmm_h != rmm_height) {
    if (this->cached_tex) {
      cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(this->cached_tex));
      this->cached_tex = 0;
    }
    cudaChannelFormatDesc channelDesc;
#if FULL_DOUBLE
    channelDesc = cudaCreateChannelDesc<int2>();
#else
    channelDesc = cudaCreateChannelDesc<float>();
#endif
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = src;
    resDesc.res.pitch2D.desc   = channelDesc;
    resDesc.res.pitch2D.width  = rmm_width;
    resDesc.res.pitch2D.height = rmm_height;
    resDesc.res.pitch2D.pitchInBytes = sizeof(scalar_type) * rmm_width;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    this->cached_tex     = static_cast<unsigned long long>(tex);
    this->cached_tex_src = src;
    this->cached_rmm_w = rmm_width;
    this->cached_rmm_h = rmm_height;
  }
  cudaTextureObject_t rmm_input_gpu_tex =
      static_cast<cudaTextureObject_t>(this->cached_tex);

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
    CudaMatrix<scalar_type>& energy_gpu = this->energy_cached;
    energy_gpu.resize(this->number_of_points);

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
    // The gather kernel wrote a symmetric buffer, so the cuArray copy made
    // above is already in the form density_derivs expects — no re-pack /
    // re-upload needed.

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

    CudaMatrix<scalar_type>& rmm_output_gpu = this->rmm_output_cached;
    rmm_output_gpu.resize(COALESCED_DIMENSION(group_m), group_m);
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
      static const char* off = getenv("LIO_RMM_CUBLAS");
      if (off && off[0] == '0') {
        gpu_update_rmm<scalar_type,true><<<threadGrid, threadBlock>>>(factors_gpu.data, this->number_of_points,
                                                                      rmm_output_gpu.data, function_values.data,
                                                                      group_m);
      } else {
        gpu_update_rmm_cublas<scalar_type>(factors_gpu.data, this->number_of_points,
                                           rmm_output_gpu.data, function_values.data,
                                           group_m, COALESCED_DIMENSION(this->number_of_points),
                                           COALESCED_DIMENSION(group_m),
                                           this->rmm_scaled_scratch);
      }
    } else {
        gpu_update_rmm<scalar_type,false><<<threadGrid, threadBlock>>>(factors_gpu.data, this->number_of_points,
                                                                       rmm_output_gpu.data, function_values.data,
                                                                       group_m);
    }

    cudaAssertNoError("update_rmm");

    /*** Scatter local Fock to global packed Fock on GPU. Index tables
     *** (bigs/rows/cols) were already uploaded by the gather block above. */
    if (n_indexes > 0) {
      dim3 scatter_block(256);
      dim3 scatter_grid((n_indexes + 255) / 256);
      gpu_scatter_rmm<scalar_type><<<scatter_grid, scatter_block>>>(
          rmm_output_gpu.data, this->rmm_bigs_gpu.data,
          this->rmm_rows_gpu.data, this->rmm_cols_gpu.data,
          s_global_fock_dev.data, n_indexes, rmm_width);
      cudaAssertNoError("gpu_scatter_rmm");
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

  // Reuse per-group cached scratch (closed-shell mirror — see solve_closed).
  // Sizes are fixed per group; the resize() calls below are no-ops after
  // the first iteration.
  CudaMatrix<scalar_type>& factors_a_gpu = this->factors_a_cached;
  CudaMatrix<scalar_type>& factors_b_gpu = this->factors_b_cached;

  // Gradients (dxyz) and Hessians (dd1,dd2) for alpha/beta.
  CudaMatrix<scalar_type>&              partial_densities_a_gpu = this->partial_densities_a_cached;
  CudaMatrix<vec_type<scalar_type,4> >& dxyz_a_gpu              = this->dxyz_a_cached;
  CudaMatrix<vec_type<scalar_type,4> >& dd1_a_gpu               = this->dd1_a_cached;
  CudaMatrix<vec_type<scalar_type,4> >& dd2_a_gpu               = this->dd2_a_cached;

  CudaMatrix<scalar_type>&              partial_densities_b_gpu = this->partial_densities_b_cached;
  CudaMatrix<vec_type<scalar_type,4> >& dxyz_b_gpu              = this->dxyz_b_cached;
  CudaMatrix<vec_type<scalar_type,4> >& dd1_b_gpu               = this->dd1_b_cached;
  CudaMatrix<vec_type<scalar_type,4> >& dd2_b_gpu               = this->dd2_b_cached;

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

  // GPU-side fused alpha+beta gather from the once-per-solve global P
  // buffers (see s_global_rdm_a/b_dev). Writes the symmetric (group_m ×
  // group_m) block of each spin's local scratch; the cuArrays below get fed
  // via D2D copies. Replaces the per-group CPU pack + 2 Host→Array uploads.
  const unsigned int rmm_width = COALESCED_DIMENSION(group_m);
  const unsigned int rmm_height = group_m + DENSITY_BLOCK_SIZE;
  CudaMatrix<scalar_type>& rdm_local_dev_a = this->rdm_local_dev_a_cached;
  CudaMatrix<scalar_type>& rdm_local_dev_b = this->rdm_local_dev_b_cached;
  if (!rdm_local_dev_a.is_allocated() ||
      rdm_local_dev_a.width != rmm_width || rdm_local_dev_a.height != rmm_height) {
    rdm_local_dev_a.resize(rmm_width, rmm_height);
    cudaMemsetAsync(rdm_local_dev_a.data, 0,
                    sizeof(scalar_type) * rmm_width * rmm_height, 0);
  }
  if (!rdm_local_dev_b.is_allocated() ||
      rdm_local_dev_b.width != rmm_width || rdm_local_dev_b.height != rmm_height) {
    rdm_local_dev_b.resize(rmm_width, rmm_height);
    cudaMemsetAsync(rdm_local_dev_b.data, 0,
                    sizeof(scalar_type) * rmm_width * rmm_height, 0);
  }
  const unsigned int n_indexes = this->rmm_bigs.size();
  if (!this->rmm_bigs_gpu.is_allocated() && n_indexes > 0) {
    this->rmm_bigs_gpu.resize(n_indexes, 1);
    this->rmm_rows_gpu.resize(n_indexes, 1);
    this->rmm_cols_gpu.resize(n_indexes, 1);
    cudaMemcpy(this->rmm_bigs_gpu.data, this->rmm_bigs.data(),
               n_indexes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->rmm_rows_gpu.data, this->rmm_rows.data(),
               n_indexes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->rmm_cols_gpu.data, this->rmm_cols.data(),
               n_indexes * sizeof(unsigned int), cudaMemcpyHostToDevice);
  }
  if (n_indexes > 0) {
    dim3 gather_block(256);
    dim3 gather_grid((n_indexes + 255) / 256);
    gpu_gather_rdm_open<scalar_type><<<gather_grid, gather_block>>>(
        s_global_rdm_a_dev.data, s_global_rdm_b_dev.data,
        this->rmm_bigs_gpu.data, this->rmm_rows_gpu.data,
        this->rmm_cols_gpu.data,
        rdm_local_dev_a.data, rdm_local_dev_b.data,
        n_indexes, rmm_width);
    cudaAssertNoError("gpu_gather_rdm_open");
  }

  /*
  **********************************************************************
  * Pasando RDM (rmm) a texturas/
  **********************************************************************
  */

  // Bind cudaTextureObject_ts directly to the gather output buffers via
  // cudaResourceTypePitch2D. Rebuilt only when a buffer pointer or the
  // dimensions change (cudaMallocAsync may return a new ptr on resize).
  void* src_a = static_cast<void*>(rdm_local_dev_a.data);
  void* src_b = static_cast<void*>(rdm_local_dev_b.data);
  if (this->cached_tex == 0 || this->cached_tex_b == 0 ||
      this->cached_tex_src   != src_a ||
      this->cached_tex_src_b != src_b ||
      this->cached_rmm_w != rmm_width ||
      this->cached_rmm_h != rmm_height) {
    if (this->cached_tex) {
      cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(this->cached_tex));
      this->cached_tex = 0;
    }
    if (this->cached_tex_b) {
      cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(this->cached_tex_b));
      this->cached_tex_b = 0;
    }

    cudaChannelFormatDesc channelDesc;
#if FULL_DOUBLE
    channelDesc = cudaCreateChannelDesc<int2>();
#else
    channelDesc = cudaCreateChannelDesc<float>();
#endif
    cudaResourceDesc resDesc1{}, resDesc2{};
    resDesc1.resType = cudaResourceTypePitch2D;
    resDesc1.res.pitch2D.devPtr = src_a;
    resDesc1.res.pitch2D.desc   = channelDesc;
    resDesc1.res.pitch2D.width  = rmm_width;
    resDesc1.res.pitch2D.height = rmm_height;
    resDesc1.res.pitch2D.pitchInBytes = sizeof(scalar_type) * rmm_width;
    resDesc2 = resDesc1;
    resDesc2.res.pitch2D.devPtr = src_b;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex_a = 0, tex_b = 0;
    cudaCreateTextureObject(&tex_a, &resDesc1, &texDesc, NULL);
    cudaCreateTextureObject(&tex_b, &resDesc2, &texDesc, NULL);
    this->cached_tex       = static_cast<unsigned long long>(tex_a);
    this->cached_tex_b     = static_cast<unsigned long long>(tex_b);
    this->cached_tex_src   = src_a;
    this->cached_tex_src_b = src_b;
    this->cached_rmm_w = rmm_width;
    this->cached_rmm_h = rmm_height;
  }
  cudaTextureObject_t rmm_input_gpu_tex  =
      static_cast<cudaTextureObject_t>(this->cached_tex);
  cudaTextureObject_t rmm_input_gpu_tex2 =
      static_cast<cudaTextureObject_t>(this->cached_tex_b);

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
    CudaMatrix<scalar_type>& energy_gpu = this->energy_cached;
    energy_gpu.resize(this->number_of_points);

    if (compute_forces || compute_rmm) {
      if (group_m <= (uint)DENSITY_BLOCK_SIZE) {
        gpu_compute_density_opened<scalar_type, false, true><<<threadGrid, threadBlock>>>(
             rmm_input_gpu_tex, rmm_input_gpu_tex2,
             this->number_of_points, function_values_transposed_ptr,
             gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
             partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
             partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
      } else {
        gpu_compute_density_opened<scalar_type, false, false><<<threadGrid, threadBlock>>>(
             rmm_input_gpu_tex, rmm_input_gpu_tex2,
             this->number_of_points, function_values_transposed_ptr,
             gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
             partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
             partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
      }
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
      if (group_m <= (uint)DENSITY_BLOCK_SIZE) {
        gpu_compute_density_opened<scalar_type, false, true><<<threadGrid, threadBlock>>>(
             rmm_input_gpu_tex, rmm_input_gpu_tex2,
             this->number_of_points, function_values_transposed_ptr,
             gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
             partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
             partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
      } else {
        gpu_compute_density_opened<scalar_type, false, false><<<threadGrid, threadBlock>>>(
             rmm_input_gpu_tex, rmm_input_gpu_tex2,
             this->number_of_points, function_values_transposed_ptr,
             gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
             partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
             partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
      }
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
    if (group_m <= (uint)DENSITY_BLOCK_SIZE) {
      gpu_compute_density_opened<scalar_type, false, true><<<threadGrid, threadBlock>>>(
           rmm_input_gpu_tex, rmm_input_gpu_tex2,
           this->number_of_points, function_values_transposed_ptr,
           gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
           partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
           partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
    } else {
      gpu_compute_density_opened<scalar_type, false, false><<<threadGrid, threadBlock>>>(
           rmm_input_gpu_tex, rmm_input_gpu_tex2,
           this->number_of_points, function_values_transposed_ptr,
           gradient_values_transposed_ptr,hessian_values_transposed.data, group_m,
           partial_densities_a_gpu.data, dxyz_a_gpu.data, dd1_a_gpu.data, dd2_a_gpu.data,
           partial_densities_b_gpu.data, dxyz_b_gpu.data, dd1_b_gpu.data, dd2_b_gpu.data);
    }
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
    // The open-shell gather kernel wrote symmetric buffers and the cuArrays
    // are populated above via D2D — no re-pack / re-upload needed for
    // density_derivs.

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

    // Combined alpha+beta output: columns [0..group_m) hold rmm_a,
    // columns [group_m..2*group_m) hold rmm_b. Letting one allocation cover
    // both spins lets us run a single fused GEMM (cuBLAS path), single fused
    // scatter, and skip the per-iter pre-zero — gpu_update_rmm and the GEMM
    // both fully write every entry the scatter ever reads (lower triangle).
    const unsigned int ldc_out = COALESCED_DIMENSION(group_m);
    CudaMatrix<scalar_type>& rmm_output_ab_gpu = this->rmm_output_ab_cached;
    rmm_output_ab_gpu.resize(ldc_out, 2 * group_m);
    scalar_type* const rmm_out_a_ptr = rmm_output_ab_gpu.data;
    scalar_type* const rmm_out_b_ptr =
        rmm_output_ab_gpu.data + (size_t)ldc_out * group_m;

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
      static const char* off = getenv("LIO_RMM_CUBLAS");
      if (off && off[0] == '0') {
        // Hand-kernel path retained for the env-var escape hatch. One fused
        // launch produces both alpha and beta into halves of the combined
        // buffer, sharing Fi/Fj loads.
        gpu_update_rmm_open<scalar_type,true><<<threadGrid, threadBlock>>>(
            factors_a_gpu.data, factors_b_gpu.data, this->number_of_points,
            rmm_out_a_ptr, rmm_out_b_ptr, function_values.data, group_m);
      } else {
        // Fused alpha+beta cuBLAS path: two dgmms + one larger GEMM, replacing
        // the four prior cuBLAS launches and giving the GEMM 2x the tile
        // count at our small group_m.
        gpu_update_rmm_cublas_open<scalar_type>(
            factors_a_gpu.data, factors_b_gpu.data, this->number_of_points,
            rmm_out_a_ptr, function_values.data,
            group_m, COALESCED_DIMENSION(this->number_of_points), ldc_out,
            this->rmm_scaled_scratch);
      }
    } else {
      // Single-block path (group_m <= RMM_BLOCK_SIZE_XY): too small for cuBLAS
      // launch overhead. Fused open kernel halves launches vs the prior
      // alpha+beta pair, which dominates TDDFT for small molecules where most
      // groups are cubes with group_m <= 16.
      gpu_update_rmm_open<scalar_type,false><<<threadGrid, threadBlock>>>(
          factors_a_gpu.data, factors_b_gpu.data, this->number_of_points,
          rmm_out_a_ptr, rmm_out_b_ptr, function_values.data, group_m);
    }

    cudaAssertNoError("update_rmm");

    /*** Scatter local Fock (alpha+beta) to global packed Fock on GPU. Index
     *** tables (bigs/rows/cols) were already uploaded by the gather block
     *** above. Fused scatter: one launch writes both global_fock_a and _b,
     *** sharing index loads across alpha+beta. */
    if (n_indexes > 0) {
      dim3 scatter_block(256);
      dim3 scatter_grid((n_indexes + 255) / 256);
      gpu_scatter_rmm_open<scalar_type><<<scatter_grid, scatter_block>>>(
          rmm_out_a_ptr, rmm_out_b_ptr,
          this->rmm_bigs_gpu.data,
          this->rmm_rows_gpu.data, this->rmm_cols_gpu.data,
          s_global_fock_a_dev.data, s_global_fock_b_dev.data,
          n_indexes, ldc_out);
      cudaAssertNoError("gpu_scatter_rmm_open");
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
