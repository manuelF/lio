// -*- mode: c -*-

/**
 * @file rmm_scatter.h
 * @brief CUDA kernel for scattering a group-local RMM output (Fock) matrix
 *        back into the global packed-triangular Fock matrix on the GPU.
 *
 * Replaces the CPU-side add_rmm_output() scatter loop + cudaMemcpy D2H.
 * The global Fock buffer is zeroed once per Partition::solve(); each group's
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
    const unsigned int* __restrict__ bigs,
    const unsigned int* __restrict__ rows,
    const unsigned int* __restrict__ cols,
    double* __restrict__ global_rmm_out,
    unsigned int n_indexes,
    unsigned int rmm_width) {
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_indexes) return;

  unsigned int r = rows[k];
  unsigned int c = cols[k];

  // gpu_update_rmm writes lower triangle: rmm[COALESCED_DIM(m) * j + i]
  // where i <= j.  In our index arrays, rows[k] <= cols[k], so:
  //   i = rows[k], j = cols[k]
  //   element at: local_rmm[cols[k] * rmm_width + rows[k]]
  double val = (double)local_rmm[c * rmm_width + r];

  atomicAdd(&global_rmm_out[bigs[k]], val);
}

/**
 * @brief Open-shell variant: scatter both alpha and beta local Fock submatrices
 *        into their respective global packed Fock buffers in a single kernel
 *        launch. Shares the row/col/bigs index loads across alpha+beta, halving
 *        launch overhead and dedup'ing the index table fetches.
 *
 * Both local_rmm_a and local_rmm_b share the same row/col layout (built from
 * the same group_m), so the same (rows[k], cols[k], bigs[k]) triple selects
 * the corresponding entry in each.
 */
template <class scalar_type>
__global__ void gpu_scatter_rmm_open(
    const scalar_type* __restrict__ local_rmm_a,
    const scalar_type* __restrict__ local_rmm_b,
    const unsigned int* __restrict__ bigs,
    const unsigned int* __restrict__ rows,
    const unsigned int* __restrict__ cols,
    double* __restrict__ global_rmm_a_out,
    double* __restrict__ global_rmm_b_out,
    unsigned int n_indexes,
    unsigned int rmm_width) {
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_indexes) return;

  unsigned int r = rows[k];
  unsigned int c = cols[k];
  unsigned int big = bigs[k];
  unsigned int idx = c * rmm_width + r;

  double val_a = (double)local_rmm_a[idx];
  double val_b = (double)local_rmm_b[idx];

  atomicAdd(&global_rmm_a_out[big], val_a);
  atomicAdd(&global_rmm_b_out[big], val_b);
}

#endif  // G2G_KERNELS_RMM_SCATTER_H
