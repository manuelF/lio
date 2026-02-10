// -*- mode: c -*-

/**
 * @file energy_derivs.h
 * @brief CUDA kernels for computing density derivatives with respect to nuclear positions.
 *
 * This file implements the computation of atomic forces via density derivatives.
 * It uses texture memory for the density matrix and shared memory batches for efficiency.
 */

#ifndef G2G_KERNELS_ENERGY_DERIVS_H
#define G2G_KERNELS_ENERGY_DERIVS_H

/* ==========================================================================================
 * CLOSED-SHELL DENSITY DERIVATIVES KERNEL
 * ========================================================================================== */

/**
 * @brief Computes density derivatives for atomic force calculations.
 *
 * @tparam scalar_type Precision type.
 *
 * @param[in]  rmm_input_gpu_tex Texture object for the density matrix R.
 * @param[in]  function_values  Basis function values on the grid.
 * @param[in]  gradient_values  Basis function gradients on the grid.
 * @param[in]  nuc              Nuclear index mapping for basis functions.
 * @param[out] density_deriv    Output array for density derivatives per nucleus and point.
 * @param[in]  points           Total number of grid points.
 * @param[in]  m                Total number of basis functions.
 * @param[in]  nuc_count        Number of nuclei in the system.
 */
template <class scalar_type>
__global__ void gpu_compute_density_derivs(
    cudaTextureObject_t rmm_input_gpu_tex, scalar_type* function_values,
    vec_type<scalar_type, 4>* gradient_values, uint* nuc,
    vec_type<scalar_type, 4>* density_deriv, uint points, uint m,
    uint nuc_count) {

  uint point = index_x(blockDim, blockIdx, threadIdx);
  bool valid_thread = (point < points);

  // Shared memory buffers for batch processing of basis function pairs
  __shared__ scalar_type rdm_sh[DENSITY_DERIV_BATCH_SIZE];
  __shared__ uint nuc_sh[DENSITY_DERIV_BATCH_SIZE2];

  // Loop over basis functions in batches
  for (uint bi = 0; bi < m; bi += DENSITY_DERIV_BATCH_SIZE2) {
    
    // 1. Parallel load of nuclear indices for the current batch
    __syncthreads();
    if (threadIdx.x < DENSITY_DERIV_BATCH_SIZE2) {
      if (bi + threadIdx.x < m) nuc_sh[threadIdx.x] = nuc[bi + threadIdx.x];
    }
    __syncthreads();

    // 2. Iterate through functions in the current batch
    for (uint i = 0; i < DENSITY_DERIV_BATCH_SIZE2 && (i + bi) < m; i++) {
      scalar_type w = 0.0f;

      // 3. Inner loop over all basis functions (j) to compute the contraction w_i = sum_j R_ij * phi_j
      for (uint bj = 0; bj < m; bj += DENSITY_DERIV_BATCH_SIZE) {
        
        // Load a batch of density matrix elements R_ij into shared memory
        __syncthreads();
        if (threadIdx.x < DENSITY_DERIV_BATCH_SIZE) {
          if (bj + threadIdx.x < m)
            rdm_sh[threadIdx.x] = fetch(rmm_input_gpu_tex, (float)(bi + i),
                                        (float)(bj + threadIdx.x));
          else
            rdm_sh[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Contract with basis function values
        if (valid_thread) {
          for (uint j = 0; j < DENSITY_DERIV_BATCH_SIZE && (bj + j) < m; j++) {
            scalar_type fj = function_values[COALESCED_DIMENSION(points) * (bj + j) + point];
            // Symmetry factor: Diagonal elements are counted once, off-diagonal twice
            w += rdm_sh[j] * fj * ((bi + i) == (bj + j) ? 2 : 1);
          }
        }
      }

      // 4. Update the density derivative for the nucleus associated with phi_i
      if (valid_thread) {
        vec_type<scalar_type, 4> Fgi = gradient_values[COALESCED_DIMENSION(points) * (bi + i) + point];
        uint nuci = nuc_sh[i];
        density_deriv[COALESCED_DIMENSION(points) * nuci + point] -= Fgi * w;
      }
    }
  }
}

/* ==========================================================================================
 * OPEN-SHELL DENSITY DERIVATIVES KERNEL
 * ========================================================================================== */

/**
 * @brief Computes open-shell density derivatives for atomic force calculations.
 *
 * Processes both alpha and beta spin components simultaneously.
 */
template <class scalar_type>
__global__ void gpu_compute_density_derivs_open(
    cudaTextureObject_t rmm_input_gpu_tex,
    cudaTextureObject_t rmm_input_gpu_tex2, scalar_type* function_values,
    vec_type<scalar_type, 4>* gradient_values, uint* nuc,
    vec_type<scalar_type, 4>* density_deriv_a,
    vec_type<scalar_type, 4>* density_deriv_b, uint points, uint m,
    uint nuc_count) {

  uint point = index_x(blockDim, blockIdx, threadIdx);
  bool valid_thread = (point < points);

  __shared__ scalar_type rdm_a_sh[DENSITY_DERIV_BATCH_SIZE];
  __shared__ scalar_type rdm_b_sh[DENSITY_DERIV_BATCH_SIZE];
  __shared__ uint nuc_sh[DENSITY_DERIV_BATCH_SIZE2];

  for (uint bi = 0; bi < m; bi += DENSITY_DERIV_BATCH_SIZE2) {
    __syncthreads();
    if (threadIdx.x < DENSITY_DERIV_BATCH_SIZE2) {
      if (bi + threadIdx.x < m) nuc_sh[threadIdx.x] = nuc[bi + threadIdx.x];
    }
    __syncthreads();

    for (uint i = 0; i < DENSITY_DERIV_BATCH_SIZE2 && (i + bi) < m; i++) {
      vec_type<scalar_type, 4> Fgi;
      if (valid_thread)
        Fgi = gradient_values[COALESCED_DIMENSION(points) * (bi + i) + point];
        
      scalar_type w_a = 0.0f;
      scalar_type w_b = 0.0f;

      for (uint bj = 0; bj < m; bj += DENSITY_DERIV_BATCH_SIZE) {
        __syncthreads();
        if (threadIdx.x < DENSITY_DERIV_BATCH_SIZE) {
          if (bj + threadIdx.x < m) {
            rdm_a_sh[threadIdx.x] = fetch(rmm_input_gpu_tex, (float)(bi + i),
                                          (float)(bj + threadIdx.x));
            rdm_b_sh[threadIdx.x] = fetch(rmm_input_gpu_tex2, (float)(bi + i),
                                          (float)(bj + threadIdx.x));
          } else {
            rdm_a_sh[threadIdx.x] = 0.0f;
            rdm_b_sh[threadIdx.x] = 0.0f;
          }
        }
        __syncthreads();

        if (valid_thread) {
          for (uint j = 0; j < DENSITY_DERIV_BATCH_SIZE && (bj + j) < m; j++) {
            scalar_type fj = function_values[COALESCED_DIMENSION(points) * (bj + j) + point];
            scalar_type factor = ((bi + i) == (bj + j) ? 2 : 1);
            w_a += rdm_a_sh[j] * fj * factor;
            w_b += rdm_b_sh[j] * fj * factor;
          }
        }
      }

      if (valid_thread) {
        uint nuci = nuc_sh[i];
        density_deriv_a[COALESCED_DIMENSION(points) * nuci + point] -= Fgi * w_a;
        density_deriv_b[COALESCED_DIMENSION(points) * nuci + point] -= Fgi * w_b;
      }
    }
  }
}

#endif // G2G_KERNELS_ENERGY_DERIVS_H
