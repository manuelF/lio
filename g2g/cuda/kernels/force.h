// -*- mode: c -*-

/**
 * @file force.h
 * @brief CUDA kernels for computing atomic forces from density derivatives and
 * XC factors.
 */

#ifndef G2G_KERNELS_FORCE_H
#define G2G_KERNELS_FORCE_H

/* ==========================================================================================
 * CLOSED-SHELL FORCE KERNEL
 * ==========================================================================================
 */

/**
 * @brief Computes atomic forces for closed-shell systems.
 *
 * This kernel integrates the product of density derivatives and XC force
 * factors over the numerical grid for each atom.
 *
 * @tparam scalar_type Precision type.
 *
 * @param[in]  points        Total number of grid points.
 * @param[in]  force_factors XC force factors at each grid point.
 * @param[in]  density_deriv Density derivatives per nucleus and point.
 * @param[out] forces        Accumulated forces for each atom.
 * @param[in]  nucleii_count Total number of nuclei.
 */
template <class scalar_type>
__global__ void gpu_compute_forces(uint points, scalar_type* force_factors,
                                   vec_type<scalar_type, 4>* density_deriv,
                                   vec_type<scalar_type, 4>* forces,
                                   uint nucleii_count) {
  uint atom = index_x(blockDim, blockIdx, threadIdx);
  bool valid_thread = (atom < nucleii_count);

  vec_type<scalar_type, 4> atom_force(0.0f, 0.0f, 0.0f, 0.0f);

  // Shared memory to cache force factors for efficient reused across atoms
  __shared__ scalar_type factor_sh[FORCE_BLOCK_SIZE];

  // Loop over grid points in chunks
  for (uint point_base = 0; point_base < points;
       point_base += FORCE_BLOCK_SIZE) {
    // 1. Parallel load of force factors into shared memory
    __syncthreads();
    if (point_base + threadIdx.x < points)
      factor_sh[threadIdx.x] = force_factors[point_base + threadIdx.x];
    __syncthreads();

    // 2. Accumulate force contribution for current atom
    if (valid_thread) {
      for (uint point_sub = 0;
           point_sub < FORCE_BLOCK_SIZE && (point_base + point_sub < points);
           point_sub++) {
        // Access density derivatives (Note: this is uncoalesced as it's indexed
        // by atom then point)
        vec_type<scalar_type, 4> density_deriv_local =
            density_deriv[COALESCED_DIMENSION(points) * atom +
                          (point_base + point_sub)];
        atom_force += density_deriv_local * factor_sh[point_sub];
      }
    }
  }

  // 3. Write final force for the atom
  if (valid_thread) {
    forces[atom] = atom_force;
  }
}

/* ==========================================================================================
 * OPEN-SHELL FORCE KERNEL
 * ==========================================================================================
 */

/**
 * @brief Computes atomic forces for open-shell systems.
 */
template <class scalar_type>
__global__ void gpu_compute_forces_open(
    uint points, scalar_type* force_factors_a, scalar_type* force_factors_b,
    vec_type<scalar_type, 4>* density_deriv_a,
    vec_type<scalar_type, 4>* density_deriv_b,
    vec_type<scalar_type, 4>* forces_a, vec_type<scalar_type, 4>* forces_b,
    uint nucleii_count) {
  uint atom = index_x(blockDim, blockIdx, threadIdx);
  bool valid_thread = (atom < nucleii_count);

  vec_type<scalar_type, 4> atom_force_a(0.0f, 0.0f, 0.0f, 0.0f);
  vec_type<scalar_type, 4> atom_force_b(0.0f, 0.0f, 0.0f, 0.0f);

  __shared__ scalar_type factor_a_sh[FORCE_BLOCK_SIZE];
  __shared__ scalar_type factor_b_sh[FORCE_BLOCK_SIZE];

  for (uint point_base = 0; point_base < points;
       point_base += FORCE_BLOCK_SIZE) {
    __syncthreads();
    if (point_base + threadIdx.x < points) {
      factor_a_sh[threadIdx.x] = force_factors_a[point_base + threadIdx.x];
      factor_b_sh[threadIdx.x] = force_factors_b[point_base + threadIdx.x];
    }
    __syncthreads();

    if (valid_thread) {
      for (uint point_sub = 0;
           point_sub < FORCE_BLOCK_SIZE && (point_base + point_sub < points);
           point_sub++) {
        uint deriv_idx =
            COALESCED_DIMENSION(points) * atom + (point_base + point_sub);

        vec_type<scalar_type, 4> deriv_a = density_deriv_a[deriv_idx];
        vec_type<scalar_type, 4> deriv_b = density_deriv_b[deriv_idx];

        atom_force_a += deriv_a * factor_a_sh[point_sub];
        atom_force_b += deriv_b * factor_b_sh[point_sub];
      }
    }
  }

  if (valid_thread) {
    forces_a[atom] = atom_force_a;
    forces_b[atom] = atom_force_b;
  }
}

#endif  // G2G_KERNELS_FORCE_H