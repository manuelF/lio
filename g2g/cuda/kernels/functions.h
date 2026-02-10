/**
 * @file functions.h
 * @brief CUDA kernels for evaluating basis functions, gradients, and Hessians
 * on a grid.
 *
 * This file implements the evaluation of Gaussian-type orbitals (GTOs) for S,
 * P, and D shells. It uses a contracted Gaussian basis set where each basis
 * function is a linear combination of primitive Gaussians.
 */

/* ==========================================================================================
 * RADIAL PART COMPUTATION
 * ==========================================================================================
 */

/**
 * @brief Computes the radial part of a contracted Gaussian basis function.
 *
 * This function calculates the value (t), the gradient factor (tg), and the
 * Hessian factor (th) for a basis function at a specific grid point.
 *
 * @tparam scalar_type Precision type (float or double).
 * @tparam do_forces   Whether to compute gradient factors.
 * @tparam do_gga      Whether to compute Hessian factors.
 *
 * @param[in] nuc            Index of the nucleus where the basis function is
 * centered.
 * @param[in] point_position Coordinates of the grid point.
 * @param[in] contractions   Number of primitive Gaussians in the contraction.
 * @param[in] factor_a_sh    Shared memory pointer to primitive exponents
 * (alpha).
 * @param[in] factor_c_sh    Shared memory pointer to contraction coefficients.
 * @param[out] t             Evaluated radial value.
 * @param[out] tg            Evaluated radial gradient factor.
 * @param[out] th            Evaluated radial Hessian factor.
 * @param[out] v             Vector from nucleus to grid point (r - R_I).
 */
template <class scalar_type, bool do_forces, bool do_gga>
static __device__ __host__ void compute_radial_factors(
    uint nuc, vec_type<scalar_type, 3> point_position, uint contractions,
    const scalar_type* factor_a_sh, const scalar_type* factor_c_sh,
    scalar_type& t, scalar_type& tg, scalar_type& th,
    vec_type<scalar_type, 3>& v) {
  vec_type<scalar_type, 3> atom_nuc_position(gpu_atom_positions[nuc]);
  v = point_position - atom_nuc_position;
  scalar_type dist2 = length2(v);

  t = 0.0f;
  if (do_forces || do_gga) tg = 0.0f;
  if (do_gga) th = 0.0f;

  for (uint contraction = 0; contraction < contractions; contraction++) {
    scalar_type a = factor_a_sh[contraction];
    scalar_type exponent = a * dist2;

    /**
     * Optimization: Skip negligible contributions.
     * e^(-70) is approximately 10^-31, which is near the limits of single
     * precision.
     */
    if (exponent > 70.0) continue;

    scalar_type t0 = exp(-exponent) * factor_c_sh[contraction];
    t += t0;

    if (do_forces || do_gga) tg += t0 * a;
    if (do_gga) th += t0 * (a * a);
  }
}

/* ==========================================================================================
 * SHELL-SPECIFIC EVALUATIONS
 * ==========================================================================================
 */

/**
 * @brief Evaluates S-type basis functions (l=0).
 *
 * phi = t
 */
template <class scalar_type, bool do_forces, bool do_gga>
static __device__ void eval_s_shell(
    uint idx, uint hidx1, uint hidx2, scalar_type t, scalar_type tg,
    scalar_type th, const vec_type<scalar_type, 3>& v,
    const vec_type<scalar_type, 3>& vxxy, const vec_type<scalar_type, 3>& vyzz,
    scalar_type* function_values, vec_type<scalar_type, 4>* gradient_values,
    vec_type<scalar_type, 4>* hessian_values) {
  function_values[idx] = t;

  if (do_forces || do_gga)
    gradient_values[idx] = vec_type<scalar_type, 4>(v * (-2.0f * tg));

  if (do_gga) {
    // Fxx, Fxy, Fxz components
    hessian_values[hidx1] =
        vec_type<scalar_type, 4>((v * v) * 4.0f * th - 2.0f * tg);
    // Fxy, Fxz, Fyz components (cross terms)
    hessian_values[hidx2] = vec_type<scalar_type, 4>(vxxy * vyzz * 4.0f * th);
  }
}

/**
 * @brief Evaluates P-type basis functions (l=1): Px, Py, Pz.
 *
 * phi = v_i * t
 */
template <class scalar_type, bool do_forces, bool do_gga>
static __device__ void eval_p_shell(
    uint idx, uint hidx1, uint hidx2, uint p_idx, scalar_type t, scalar_type tg,
    scalar_type th, const vec_type<scalar_type, 3>& v,
    const vec_type<scalar_type, 3>& vxxy, const vec_type<scalar_type, 3>& vyzz,
    scalar_type* function_values, vec_type<scalar_type, 4>* gradient_values,
    vec_type<scalar_type, 4>* hessian_values) {
  scalar_type v_comp;
  vec_type<scalar_type, 3> grad_base;
  vec_type<scalar_type, 3> hess_coeff;
  vec_type<scalar_type, 3> hess_cross;

  // Select components based on Px, Py, or Pz
  switch (p_idx) {
    case 0:  // Px
      v_comp = v.x;
      grad_base = vec_type<scalar_type, 3>(t, 0.0f, 0.0f);
      hess_coeff = vec_type<scalar_type, 3>(6.0f, 2.0f, 2.0f);
      hess_cross = vec_type<scalar_type, 3>(v.y, v.z, 0.0f);
      break;
    case 1:  // Py
      v_comp = v.y;
      grad_base = vec_type<scalar_type, 3>(0.0f, t, 0.0f);
      hess_coeff = vec_type<scalar_type, 3>(2.0f, 6.0f, 2.0f);
      hess_cross = vec_type<scalar_type, 3>(v.x, 0.0f, v.z);
      break;
    case 2:  // Pz
      v_comp = v.z;
      grad_base = vec_type<scalar_type, 3>(0.0f, 0.0f, t);
      hess_coeff = vec_type<scalar_type, 3>(2.0f, 2.0f, 6.0f);
      hess_cross = vec_type<scalar_type, 3>(0.0f, v.x, v.y);
      break;
  }

  function_values[idx] = v_comp * t;

  if (do_forces || do_gga) {
    gradient_values[idx] =
        vec_type<scalar_type, 4>(grad_base - v * 2.0f * tg * v_comp);
  }

  if (do_gga) {
    hessian_values[hidx1] = vec_type<scalar_type, 4>(
        (v * v) * 4.0f * th * v_comp - hess_coeff * tg * v_comp);
    hessian_values[hidx2] = vec_type<scalar_type, 4>(
        (vxxy * vyzz) * 4.0f * th * v_comp - hess_cross * 2.0f * tg);
  }
}

/**
 * @brief Evaluates D-type basis functions (l=2): XX, XY, YY, XZ, YZ, ZZ.
 *
 * phi = v_i * v_j * t
 */
template <class scalar_type, bool do_forces, bool do_gga>
static __device__ void eval_d_shell(
    uint idx, uint hidx1, uint hidx2, uint d_idx, scalar_type t, scalar_type tg,
    scalar_type th, const vec_type<scalar_type, 3>& v,
    const vec_type<scalar_type, 3>& vxxy, const vec_type<scalar_type, 3>& vyzz,
    scalar_type* function_values, vec_type<scalar_type, 4>* gradient_values,
    vec_type<scalar_type, 4>* hessian_values) {
  // Cartesian D-shell ordering:
  // 0: XX, 1: YX, 2: YY, 3: ZX, 4: ZY, 5: ZZ
  switch (d_idx) {
    case 0:  // XX
      function_values[idx] = t * v.x * v.x * gpu_normalization_factor;
      if (do_forces || do_gga)
        gradient_values[idx] = vec_type<scalar_type, 4>(
            (vec_type<scalar_type, 3>(2.0f * v.x, 0.0f, 0.0f) * t -
             v * 2.0f * tg * v.x * v.x) *
            gpu_normalization_factor);
      if (do_gga) {
        hessian_values[hidx1] = vec_type<scalar_type, 4>(
            ((v * v) * 4.0f * th * (v.x * v.x) -
             vec_type<scalar_type, 3>(10.0f, 2.0f, 2.0f) * tg * (v.x * v.x) +
             vec_type<scalar_type, 3>(2.0f * t, 0.0f, 0.0f)) *
            gpu_normalization_factor);
        hessian_values[hidx2] = vec_type<scalar_type, 4>(
            ((vxxy * vyzz) * 4.0f * th * (v.x * v.x) -
             vec_type<scalar_type, 3>(4.0f, 4.0f, 0.0f) * (vxxy * vyzz) * tg) *
            gpu_normalization_factor);
      }
      break;
    case 1:  // XY (YX)
      function_values[idx] = t * v.y * v.x;
      if (do_forces || do_gga)
        gradient_values[idx] = vec_type<scalar_type, 4>(
            vec_type<scalar_type, 3>(v.y, v.x, 0.0f) * t -
            v * 2.0f * tg * v.y * v.x);
      if (do_gga) {
        hessian_values[hidx1] = vec_type<scalar_type, 4>(
            ((v * v) * 4.0f * th * (v.x * v.y) -
             vec_type<scalar_type, 3>(6.0f, 6.0f, 2.0f) * tg * (v.x * v.y)));
        hessian_values[hidx2] = vec_type<scalar_type, 4>(
            ((vxxy * vyzz) * 4.0f * th * (v.x * v.y) -
             vec_type<scalar_type, 3>(2.0f * (v.x * v.x + v.y * v.y),
                                      2.0f * v.y * v.z, 2.0f * v.x * v.z) *
                 tg +
             vec_type<scalar_type, 3>(t, 0.0f, 0.0f)));
      }
      break;
    case 2:  // YY
      function_values[idx] = t * v.y * v.y * gpu_normalization_factor;
      if (do_forces || do_gga)
        gradient_values[idx] = vec_type<scalar_type, 4>(
            (vec_type<scalar_type, 3>(0.0f, 2.0f * v.y, 0.0f) * t -
             v * 2.0f * tg * v.y * v.y) *
            gpu_normalization_factor);
      if (do_gga) {
        hessian_values[hidx1] = vec_type<scalar_type, 4>(
            ((v * v) * 4.0f * th * (v.y * v.y) -
             vec_type<scalar_type, 3>(2.0f, 10.0f, 2.0f) * tg * (v.y * v.y) +
             vec_type<scalar_type, 3>(0.0f, 2.0f * t, 0.0f)) *
            gpu_normalization_factor);
        hessian_values[hidx2] = vec_type<scalar_type, 4>(
            ((vxxy * vyzz) * 4.0f * th * (v.y * v.y) -
             vec_type<scalar_type, 3>(4.0f, 0.0f, 4.0f) * (vxxy * vyzz) * tg) *
            gpu_normalization_factor);
      }
      break;
    case 3:  // XZ (ZX)
      function_values[idx] = t * v.z * v.x;
      if (do_forces || do_gga)
        gradient_values[idx] = vec_type<scalar_type, 4>(
            vec_type<scalar_type, 3>(v.z, 0.0f, v.x) * t -
            v * 2.0f * tg * v.z * v.x);
      if (do_gga) {
        hessian_values[hidx1] = vec_type<scalar_type, 4>(
            ((v * v) * 4.0f * th * (v.x * v.z) -
             vec_type<scalar_type, 3>(6.0f, 2.0f, 6.0f) * tg * (v.x * v.z)));
        hessian_values[hidx2] = vec_type<scalar_type, 4>(
            ((vxxy * vyzz) * 4.0f * th * (v.x * v.z) -
             vec_type<scalar_type, 3>(2.0f * v.y * v.z,
                                      2.0f * (v.x * v.x + v.z * v.z),
                                      2.0f * v.x * v.y) *
                 tg +
             vec_type<scalar_type, 3>(0.0f, t, 0.0f)));
      }
      break;
    case 4:  // YZ (ZY)
      function_values[idx] = t * v.z * v.y;
      if (do_forces || do_gga)
        gradient_values[idx] = vec_type<scalar_type, 4>(
            vec_type<scalar_type, 3>(0.0f, v.z, v.y) * t -
            v * 2.0f * tg * v.z * v.y);
      if (do_gga) {
        hessian_values[hidx1] = vec_type<scalar_type, 4>(
            ((v * v) * 4.0f * th * (v.y * v.z) -
             vec_type<scalar_type, 3>(2.0f, 6.0f, 6.0f) * tg * (v.y * v.z)));
        hessian_values[hidx2] = vec_type<scalar_type, 4>(
            ((vxxy * vyzz) * 4.0f * th * (v.y * v.z) -
             vec_type<scalar_type, 3>(2.0f * v.x * v.z, 2.0f * v.x * v.y,
                                      2.0f * (v.y * v.y + v.z * v.z)) *
                 tg +
             vec_type<scalar_type, 3>(0.0f, 0.0f, t)));
      }
      break;
    case 5:  // ZZ
      function_values[idx] = t * v.z * v.z * gpu_normalization_factor;
      if (do_forces || do_gga)
        gradient_values[idx] = vec_type<scalar_type, 4>(
            (vec_type<scalar_type, 3>(0.0f, 0.0f, 2.0f * v.z) * t -
             v * 2.0f * tg * v.z * v.z) *
            gpu_normalization_factor);
      if (do_gga) {
        hessian_values[hidx1] = vec_type<scalar_type, 4>(
            ((v * v) * 4.0f * th * (v.z * v.z) -
             vec_type<scalar_type, 3>(2.0f, 2.0f, 10.0f) * tg * (v.z * v.z) +
             vec_type<scalar_type, 3>(0.0f, 0.0f, 2.0f * t)) *
            gpu_normalization_factor);
        hessian_values[hidx2] = vec_type<scalar_type, 4>(
            ((vxxy * vyzz) * 4.0f * th * (v.z * v.z) -
             (vxxy * vyzz) * vec_type<scalar_type, 3>(0.0f, 4.0f, 4.0f) * tg) *
            gpu_normalization_factor);
      }
      break;
  }
}

/* ==========================================================================================
 * MAIN KERNEL
 * ==========================================================================================
 */

/**
 * @brief CUDA kernel to compute basis functions and their derivatives on a
 * grid.
 *
 * This kernel processes grid points in parallel. For each point, it iterates
 * through the basis functions in blocks to utilize shared memory for basis set
 * parameters.
 *
 * @tparam scalar_type Precision type.
 * @tparam do_forces   Whether to compute gradients.
 * @tparam do_gga      Whether to compute Hessians.
 *
 * @param[in]  point_positions  Array of grid point positions (x, y, z, w).
 * @param[in]  points           Number of grid points.
 * @param[in]  contractions     Array of contraction counts per basis function.
 * @param[in]  factor_ac        Coalesced array of (alpha, coefficient) pairs.
 * @param[in]  nuc              Array of nuclear indices for each basis
 * function.
 * @param[out] function_values  Output array for basis function values.
 * @param[out] gradient_values  Output array for gradients.
 * @param[out] hessian_values  Output array for Hessians.
 * @param[in]  functions        uint4 containing (s_count, p_shell_count,
 * d_shell_count, total_count).
 */
template <class scalar_type, bool do_forces, bool do_gga>
__global__ void gpu_compute_functions(vec_type<scalar_type, 4>* point_positions,
                                      uint points, uint* contractions,
                                      vec_type<scalar_type, 2>* factor_ac,
                                      uint* nuc, scalar_type* function_values,
                                      vec_type<scalar_type, 4>* gradient_values,
                                      vec_type<scalar_type, 4>* hessian_values,
                                      uint4 functions) {
  dim3 pos = index(blockDim, blockIdx, threadIdx);
  uint point = pos.x;

  // Load grid point information
  bool valid_thread = (point < points);
  vec_type<scalar_type, 3> point_position;
  if (valid_thread) {
    point_position = vec_type<scalar_type, 3>(point_positions[point]);
  }

  scalar_type t, tg, th;
  vec_type<scalar_type, 3> v;

  // Shared memory buffers for basis set parameters to reduce global memory
  // traffic
  __shared__ uint nuc_sh[FUNCTIONS_BLOCK_SIZE];
  __shared__ uint contractions_sh[FUNCTIONS_BLOCK_SIZE];
  __shared__ scalar_type factor_a_sh[FUNCTIONS_BLOCK_SIZE][MAX_CONTRACTIONS];
  __shared__ scalar_type factor_c_sh[FUNCTIONS_BLOCK_SIZE][MAX_CONTRACTIONS];

  // Iterate over basis functions in chunks of FUNCTIONS_BLOCK_SIZE
  for (uint i = 0; i < functions.w; i += FUNCTIONS_BLOCK_SIZE) {
    // 1. Parallel load of basis function data into Shared Memory
    if (i + threadIdx.x < functions.w) {
      uint offset = i + threadIdx.x;
      nuc_sh[threadIdx.x] = nuc[offset];
      uint count = contractions[offset];
      contractions_sh[threadIdx.x] = count;

      uint data_stride = COALESCED_DIMENSION(functions.w);
      for (uint contraction = 0; contraction < count; contraction++) {
        vec_type<scalar_type, 2> fac =
            factor_ac[data_stride * contraction + offset];
        factor_a_sh[threadIdx.x][contraction] = fac.x;
        factor_c_sh[threadIdx.x][contraction] = fac.y;
      }
    }
    __syncthreads();

    // 2. Main computation loop for valid threads
    if (valid_thread) {
      for (uint ii = 0; ii < FUNCTIONS_BLOCK_SIZE && (i + ii < functions.w);
           ii++) {
        uint current_func_idx = i + ii;

        // Step A: Radial part evaluation
        compute_radial_factors<scalar_type, do_forces, do_gga>(
            nuc_sh[ii], point_position, contractions_sh[ii], factor_a_sh[ii],
            factor_c_sh[ii], t, tg, th, v);

        // Step B: Determine output indices
        uint idx = COALESCED_DIMENSION(points) * current_func_idx + point;
        uint hidx1 = 0, hidx2 = 0;
        vec_type<scalar_type, 3> vxxy, vyzz;

        if (do_gga) {
          hidx1 =
              COALESCED_DIMENSION(points) * (2 * current_func_idx + 0) + point;
          hidx2 =
              COALESCED_DIMENSION(points) * (2 * current_func_idx + 1) + point;
          vxxy = vec_type<scalar_type, 3>(v.x, v.x, v.y);
          vyzz = vec_type<scalar_type, 3>(v.y, v.z, v.z);
        }

        // Step C: Angular evaluation based on shell type (S, P, D)
        if (current_func_idx < functions.x) {
          // S-shell (l=0)
          eval_s_shell<scalar_type, do_forces, do_gga>(
              idx, hidx1, hidx2, t, tg, th, v, vxxy, vyzz, function_values,
              gradient_values, hessian_values);

        } else if (current_func_idx < (functions.x + functions.y * 3)) {
          // P-shell (l=1)
          uint p_idx = (current_func_idx - functions.x) % 3;
          eval_p_shell<scalar_type, do_forces, do_gga>(
              idx, hidx1, hidx2, p_idx, t, tg, th, v, vxxy, vyzz,
              function_values, gradient_values, hessian_values);

        } else {
          // D-shell (l=2)
          uint d_idx = (current_func_idx - functions.x - functions.y * 3) % 6;
          eval_d_shell<scalar_type, do_forces, do_gga>(
              idx, hidx1, hidx2, d_idx, t, tg, th, v, vxxy, vyzz,
              function_values, gradient_values, hessian_values);
        }
      }
    }
    __syncthreads();
  }
}
