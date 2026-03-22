// -*- mode: c -*-

/**
 * @file rmm_scatter.h
 * @brief CUDA kernel for scattering a group-local RMM output (Fock) matrix
 *        back into the global packed-triangular Fock matrix on the GPU.
 *
 * Replaces the CPU-side add_rmm_output() scatter loop + cudaMemcpy D2H.
 * The global Fock buffer is zeroed once per SCF iteration; each group's
 * scatter kernel atomicAdds its contribution.
 */

#ifndef G2G_KERNELS_RMM_SCATTER_H
#define G2G_KERNELS_RMM_SCATTER_H

/**
 * @brief Scatter a group-local Fock submatrix into the global packed Fock.
 *
 * Each thread handles one (row, col) pair from the precomputed index arrays.
 * The local RMM stores only the lower triangle (i <= j), written by
 * gpu_update_rmm as rmm[COALESCED_DIM(m) * j + i].
 *
 * Uses atomicAdd(double) which is natively supported on SM 6.0+.
 *
 * @tparam scalar_type  Precision of the local RMM (float or double).
 *
 * @param[in]  local_rmm       Group-local Fock submatrix (lower triangle,
 *                              COALESCED_DIM(m) x m layout).
 * @param[in]  bigs            Flat indices into global_rmm (size n_indexes).
 * @param[in]  rows            Local row indices (size n_indexes, rows[k] <= cols[k]).
 * @param[in]  cols            Local col indices (size n_indexes).
 * @param[out] global_rmm_out  Global packed upper-triangular Fock (double).
 *                              Must be pre-zeroed at start of iteration.
 * @param[in]  n_indexes       Number of index entries.
 * @param[in]  rmm_width       Leading dimension of local_rmm (= COALESCED_DIMENSION(group_m)).
 */
template <class scalar_type>
__global__ void gpu_scatter_rmm(
    const scalar_type* __restrict__ local_rmm,
    const uint* __restrict__ bigs,
    const uint* __restrict__ rows,
    const uint* __restrict__ cols,
    double* __restrict__ global_rmm_out,
    uint n_indexes,
    uint rmm_width) {
  uint k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_indexes) return;

  uint r = rows[k];
  uint c = cols[k];

  // gpu_update_rmm writes lower triangle: rmm[COALESCED_DIM(m) * j + i]
  // where i <= j.  In our index arrays, rows[k] <= cols[k], so:
  //   i = rows[k], j = cols[k]
  //   element at: local_rmm[cols[k] * rmm_width + rows[k]]
  double val = (double)local_rmm[c * rmm_width + r];

  atomicAdd(&global_rmm_out[bigs[k]], val);
}

#endif  // G2G_KERNELS_RMM_SCATTER_H
