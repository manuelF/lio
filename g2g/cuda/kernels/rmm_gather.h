// -*- mode: c -*-

/**
 * @file rmm_gather.h
 * @brief CUDA kernel for gathering a group-local RMM submatrix from the
 *        global packed-triangular density matrix on the GPU.
 *
 * Replaces the CPU-side get_rmm_input() gather loop + cudaMemcpy2DToArrayAsync
 * texture upload.  The global RMM is uploaded once per SCF iteration; this
 * kernel runs once per PointGroup to extract the relevant submatrix.
 */

#ifndef G2G_KERNELS_RMM_GATHER_H
#define G2G_KERNELS_RMM_GATHER_H

/**
 * @brief Gather a group-local RMM submatrix from the global packed RMM.
 *
 * Each thread handles one (row, col) pair from the precomputed index arrays.
 * The output is a full symmetric local matrix (both triangles filled) so that
 * both gpu_compute_density (reads lower triangle) and
 * gpu_compute_density_derivs (reads full matrix) can use it.
 *
 * @tparam scalar_type  Output precision (float or double).
 *
 * @param[in]  global_rmm   Global packed upper-triangular RMM (double).
 *                           Index k maps to element R[i][j] where i <= j,
 *                           k = i*M - i*(i-1)/2 + (j-i).
 * @param[in]  bigs         Flat indices into global_rmm (size n_indexes).
 * @param[in]  rows         Local row indices (size n_indexes, rows[k] <= cols[k]).
 * @param[in]  cols         Local col indices (size n_indexes).
 * @param[out] local_rmm    Output submatrix (rmm_width * height), must be
 *                           pre-zeroed (padding must be 0).
 * @param[in]  n_indexes    Number of index entries.
 * @param[in]  rmm_width    Leading dimension of local_rmm (= COALESCED_DIMENSION(group_m)).
 */
template <class scalar_type>
__global__ void gpu_gather_rmm(
    const double* __restrict__ global_rmm,
    const uint* __restrict__ bigs,
    const uint* __restrict__ rows,
    const uint* __restrict__ cols,
    scalar_type* __restrict__ local_rmm,
    uint n_indexes,
    uint rmm_width) {
  uint k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_indexes) return;

  scalar_type val = (scalar_type)global_rmm[bigs[k]];
  uint r = rows[k];
  uint c = cols[k];

  // Lower triangle: data[col * width + row]  (col >= row)
  // This matches HostMatrix(row, col) = data[col * width + row]
  // and the kernel read pattern: data[i * width + j] where i >= j
  local_rmm[c * rmm_width + r] = val;

  // Upper triangle (mirror for density_derivs which reads full matrix)
  if (r != c) {
    local_rmm[r * rmm_width + c] = val;
  }
}

#endif  // G2G_KERNELS_RMM_GATHER_H
