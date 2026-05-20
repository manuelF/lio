// -*- mode: c -*-

/**
 * @file rmm_gather.h
 * @brief CUDA kernel for gathering a group-local RMM input (density)
 *        submatrix from the global packed-triangular P matrix on the GPU.
 *
 * Replaces the per-group CPU pack (`get_rmm_input` O(group_m²) loop) and the
 * Host→Array `cudaMemcpy2DToArrayAsync` that fed the density-kernel texture.
 * The global P buffer is uploaded once per Partition::solve() via
 * `gpu_upload_global_rdm`; each group's gather runs on-device using the
 * `rmm_bigs/rows/cols` index tables that already exist for the scatter side.
 *
 * Output layout matches the legacy CPU pack: a (height × width) matrix with
 * width = COALESCED_DIMENSION(group_m), height = group_m + DENSITY_BLOCK_SIZE.
 * We write the symmetric (group_m × group_m) block via the index tables; the
 * padding rows/cols past group_m stay zeroed (one-shot memset by caller on
 * first allocation).
 */

#ifndef G2G_KERNELS_RMM_GATHER_H
#define G2G_KERNELS_RMM_GATHER_H

template <class scalar_type>
__global__ void gpu_gather_rdm(
    const double* __restrict__ global_rdm,
    const unsigned int* __restrict__ bigs,
    const unsigned int* __restrict__ rows,
    const unsigned int* __restrict__ cols,
    scalar_type* __restrict__ local_rdm,
    unsigned int n_indexes,
    unsigned int rmm_width) {
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_indexes) return;
  unsigned int r = rows[k];
  unsigned int c = cols[k];
  scalar_type val = (scalar_type)global_rdm[bigs[k]];
  // Symmetric write. The density kernel only reads the lower triangle but
  // density_derivs (forces) consumes the full symmetric matrix; writing
  // both keeps the output identical to the legacy CPU pack and avoids
  // forking on compute_forces. compute_indexes() guarantees rows[k] ≤
  // cols[k]; on the diagonal both writes land on the same address.
  local_rdm[(size_t)c * rmm_width + r] = val;  // row=c, col=r
  local_rdm[(size_t)r * rmm_width + c] = val;  // row=r, col=c
}

/**
 * @brief Open-shell variant: gather alpha and beta P submatrices in a single
 *        launch. Shares the index-table loads across both spins, halving
 *        launch overhead vs. two separate kernels.
 */
template <class scalar_type>
__global__ void gpu_gather_rdm_open(
    const double* __restrict__ global_rdm_a,
    const double* __restrict__ global_rdm_b,
    const unsigned int* __restrict__ bigs,
    const unsigned int* __restrict__ rows,
    const unsigned int* __restrict__ cols,
    scalar_type* __restrict__ local_rdm_a,
    scalar_type* __restrict__ local_rdm_b,
    unsigned int n_indexes,
    unsigned int rmm_width) {
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_indexes) return;
  unsigned int r = rows[k];
  unsigned int c = cols[k];
  unsigned int big = bigs[k];
  size_t idx_lo = (size_t)c * rmm_width + r;  // row=c, col=r (lower)
  size_t idx_up = (size_t)r * rmm_width + c;  // row=r, col=c (upper / symmetric)
  scalar_type val_a = (scalar_type)global_rdm_a[big];
  scalar_type val_b = (scalar_type)global_rdm_b[big];
  local_rdm_a[idx_lo] = val_a;
  local_rdm_a[idx_up] = val_a;
  local_rdm_b[idx_lo] = val_b;
  local_rdm_b[idx_up] = val_b;
}

#endif  // G2G_KERNELS_RMM_GATHER_H
