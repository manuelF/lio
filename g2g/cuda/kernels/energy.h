// -*- mode: c -*-

/**
 * @file energy.h
 * @brief CUDA kernels for numerical integration of the density and its
 * derivatives.
 *
 * This file implements the evaluation of the electron density rho(r) and its
 * spatial derivatives (gradients and Hessians) on a numerical grid for
 * closed-shell systems. It utilises texture memory for efficient access to the
 * density matrix (RMM) and shared memory to cache basis function values.
 *
 * Warp reductions use __shfl_down_sync (Kepler+, SM 3.5+) rather than the
 * legacy volatile-shared-memory tree, eliminating store fences and freeing
 * shared memory for occupancy improvements.
 */

#ifndef G2G_KERNELS_ENERGY_H
#define G2G_KERNELS_ENERGY_H

/* ==========================================================================================
 * TEXTURE FETCH HELPERS
 * ==========================================================================================
 */

#if FULL_DOUBLE
/**
 * @brief Helper to fetch double precision values from a 2D texture.
 * Maps the int2 texture format back to a double.
 */
static __inline__ __device__ double fetch_double(cudaTextureObject_t t, float x,
                                                 float y) {
  int2 v = tex2D<int2>(t, x, y);
  return __hiloint2double(v.y, v.x);
}
#define fetch(t, x, y) fetch_double(t, x, y)
#else
/**
 * @brief Helper to fetch single precision values from a 2D texture.
 */
#define fetch(t, x, y) tex2D<float>(t, x, y)
#endif

/* ==========================================================================================
 * REDUCTION HELPERS  (register-based, __shfl_down_sync)
 *
 * These operate on register values within a single warp (32 threads).
 * No shared memory is touched — all communication happens via warp-level
 * register exchange instructions, eliminating the volatile-memory store fences
 * of the pre-Kepler pattern.
 * ==========================================================================================
 */

/**
 * @brief Warp-level sum reduction of a scalar value.
 * Returns the total sum to lane 0; other lanes hold partial results.
 */
template <typename T>
__device__ __forceinline__ T warpReduceScalar(T val) {
  val += __shfl_down_sync(0xffffffffu, val, 16);
  val += __shfl_down_sync(0xffffffffu, val,  8);
  val += __shfl_down_sync(0xffffffffu, val,  4);
  val += __shfl_down_sync(0xffffffffu, val,  2);
  val += __shfl_down_sync(0xffffffffu, val,  1);
  return val;
}

/**
 * @brief Warp-level sum reduction of a 3D vector.
 * Returns the total sum to lane 0 in each component.
 */
template <typename T>
__device__ __forceinline__ vec_type<T, 3> warpReduceVector3(vec_type<T, 3> v) {
  v.x = warpReduceScalar(v.x);
  v.y = warpReduceScalar(v.y);
  v.z = warpReduceScalar(v.z);
  return v;
}

/* ==========================================================================================
 * CLOSED-SHELL DENSITY KERNEL
 * ==========================================================================================
 */

/**
 * @brief CUDA kernel to compute density and derivatives for closed-shell
 * systems.
 *
 * rho(p) = sum_{ij} R_ij * phi_i(p) * phi_j(p)
 *
 * Block: dim3(DENSITY_BLOCK_SIZE, 1, 1) = 64 threads = 2 warps.
 * Grid:  dim3(npoints, block_height, 1) where block_height covers all
 *        function-index slices.
 *
 * Reduction: each warp reduces its 32-thread partial sum via __shfl_down_sync.
 * Cross-warp accumulation uses shared memory for the cross-warp exchange.
 *
 * @tparam scalar_type  Precision type (float / double).
 * @tparam lda          If true, only compute density. If false, also compute
 *                      gradients and Hessians.
 */
template <class scalar_type, bool lda>
__global__
void gpu_compute_density(
    cudaTextureObject_t rmm_input_gpu_tex,
    scalar_type* __restrict__ const energy,
    scalar_type* __restrict__ const factor,
    const scalar_type* __restrict__ const point_weights, uint points,
    const scalar_type* __restrict__ function_values,
    const vec_type<scalar_type, 4>* __restrict__ gradient_values,
    const vec_type<scalar_type, 4>* __restrict__ hessian_values, uint m,
    scalar_type* __restrict__ out_partial_density,
    vec_type<scalar_type, 4>* __restrict__ out_dxyz,
    vec_type<scalar_type, 4>* __restrict__ out_dd1,
    vec_type<scalar_type, 4>* __restrict__ out_dd2) {
  uint point = blockIdx.x;
  uint i = threadIdx.x + blockIdx.y * 2 * DENSITY_BLOCK_SIZE;
  uint i2 = i + DENSITY_BLOCK_SIZE;
  uint min_i = blockIdx.y * 2 * DENSITY_BLOCK_SIZE + DENSITY_BLOCK_SIZE;

  bool valid_thread = (i < m);
  bool valid_thread2 = (i2 < m);

  scalar_type w = 0.0f, w2 = 0.0f;
  vec_type<scalar_type, 3> w3, ww1, ww2;
  vec_type<scalar_type, 3> w32, ww12, ww22;

  if (!lda) {
    w3 = ww1 = ww2 = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
    w32 = ww12 = ww22 = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
  }

  int tid = threadIdx.x;

  __shared__ scalar_type fj_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 3> fgj_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 3> fh1j_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 3> fh2j_sh[DENSITY_BLOCK_SIZE];

  if (min_i > m) min_i = min_i - DENSITY_BLOCK_SIZE;

  for (int bj = 0; bj <= (int)min_i; bj += DENSITY_BLOCK_SIZE) {
    __syncthreads();
    if (bj + tid < (int)m) {
      fj_sh[tid] = function_values[(m)*point + (bj + tid)];
      if (!lda) {
        fgj_sh[tid] =
            vec_type<scalar_type, 3>(gradient_values[(m)*point + (bj + tid)]);
        fh1j_sh[tid] = vec_type<scalar_type, 3>(
            hessian_values[(m) * 2 * point + (2 * (bj + tid) + 0)]);
        fh2j_sh[tid] = vec_type<scalar_type, 3>(
            hessian_values[(m) * 2 * point + (2 * (bj + tid) + 1)]);
      }
    }
    __syncthreads();

    if (valid_thread) {
      bool full_block = (i >= (uint)(bj + DENSITY_BLOCK_SIZE - 1));
#pragma unroll 4
      for (int j = 0; j < DENSITY_BLOCK_SIZE; j++) {
        if (full_block || (bj + j) <= (int)i) {
          scalar_type rdm = fetch(rmm_input_gpu_tex, (float)(bj + j), (float)i);
          scalar_type fj_val = fj_sh[j];
          w += rdm * fj_val;
          if (!lda) {
            w3 += vec_type<scalar_type, 3>(fgj_sh[j] * rdm);
            ww1 += vec_type<scalar_type, 3>(fh1j_sh[j] * rdm);
            ww2 += vec_type<scalar_type, 3>(fh2j_sh[j] * rdm);
          }
        }
        if (valid_thread2 && (full_block || (bj + j) <= (int)i2)) {
          scalar_type rdm2 =
              fetch(rmm_input_gpu_tex, (float)(bj + j), (float)i2);
          scalar_type fj_val = fj_sh[j];
          w2 += rdm2 * fj_val;
          if (!lda) {
            w32 += vec_type<scalar_type, 3>(fgj_sh[j] * rdm2);
            ww12 += vec_type<scalar_type, 3>(fh1j_sh[j] * rdm2);
            ww22 += vec_type<scalar_type, 3>(fh2j_sh[j] * rdm2);
          }
        }
      }
    }
  }

  // --- Per-thread local partial results ---
  scalar_type partial_rho(0.0f);
  vec_type<scalar_type, 3> dxyz(0.0f, 0.0f, 0.0f), dd1(0.0f, 0.0f, 0.0f),
      dd2(0.0f, 0.0f, 0.0f);

  if (valid_thread) {
    scalar_type Fi = function_values[(m)*point + i];
    partial_rho = Fi * w;
    if (!lda) {
      vec_type<scalar_type, 3> Fgi(gradient_values[(m)*point + i]);
      vec_type<scalar_type, 3> Fhi1(
          hessian_values[(m) * 2 * point + (2 * i + 0)]);
      vec_type<scalar_type, 3> Fhi2(
          hessian_values[(m) * 2 * point + (2 * i + 1)]);
      dxyz = Fgi * w + w3 * Fi;
      dd1 = Fgi * w3 * 2.0f + Fhi1 * w + ww1 * Fi;
      vec_type<scalar_type, 3> FgXXY(Fgi.x, Fgi.x, Fgi.y);
      dd2 = FgXXY * vec_type<scalar_type, 3>(w3.y, w3.z, w3.z) +
            vec_type<scalar_type, 3>(Fgi.y, Fgi.z, Fgi.z) *
                vec_type<scalar_type, 3>(w3.x, w3.x, w3.y) +
            Fhi2 * w + ww2 * Fi;
    }
    if (valid_thread2) {
      scalar_type Fi2 = function_values[(m)*point + i2];
      partial_rho += Fi2 * w2;
      if (!lda) {
        vec_type<scalar_type, 3> Fgi2(gradient_values[(m)*point + i2]);
        vec_type<scalar_type, 3> Fhi12(
            hessian_values[(m) * 2 * point + (2 * i2 + 0)]);
        vec_type<scalar_type, 3> Fhi22(
            hessian_values[(m) * 2 * point + (2 * i2 + 1)]);
        dxyz += Fgi2 * w2 + w32 * Fi2;
        dd1 += Fgi2 * w32 * 2.0f + Fhi12 * w2 + ww12 * Fi2;
        vec_type<scalar_type, 3> FgXXY2(Fgi2.x, Fgi2.x, Fgi2.y);
        dd2 += FgXXY2 * vec_type<scalar_type, 3>(w32.y, w32.z, w32.z) +
               vec_type<scalar_type, 3>(Fgi2.y, Fgi2.z, Fgi2.z) *
                   vec_type<scalar_type, 3>(w32.x, w32.x, w32.y) +
               Fhi22 * w2 + ww22 * Fi2;
      }
    }
  }

  // --- Two-warp reduction (numerically equivalent to the original volatile
  // pattern) ---
  int lane = tid & 31;
  int warp = tid >> 5;  // 0 or 1

  // Step 1 — all threads store partials
  fj_sh[tid] = partial_rho;
  if (!lda) {
    fgj_sh[tid] = dxyz;
    fh1j_sh[tid] = dd1;
    fh2j_sh[tid] = dd2;
  }
  __syncthreads();

  // Step 2 — warp 0: cross-warp pair then intra-warp shuffle
  if (warp == 0) {
    scalar_type rho_val = fj_sh[lane] + fj_sh[lane + 32];
    rho_val = warpReduceScalar(rho_val);
    if (!lda) {
      vec_type<scalar_type, 3> dxyz_val = fgj_sh[lane] + fgj_sh[lane + 32];
      vec_type<scalar_type, 3> dd1_val = fh1j_sh[lane] + fh1j_sh[lane + 32];
      vec_type<scalar_type, 3> dd2_val = fh2j_sh[lane] + fh2j_sh[lane + 32];
      dxyz_val = warpReduceVector3(dxyz_val);
      dd1_val = warpReduceVector3(dd1_val);
      dd2_val = warpReduceVector3(dd2_val);
      // Step 3 — lane 0 writes
      if (lane == 0) {
        const int myPoint = blockIdx.y * points + blockIdx.x;
        out_partial_density[myPoint] = rho_val;
        out_dxyz[myPoint] = vec_type<scalar_type, 4>(dxyz_val);
        out_dd1[myPoint] = vec_type<scalar_type, 4>(dd1_val);
        out_dd2[myPoint] = vec_type<scalar_type, 4>(dd2_val);
      }
    } else {
      if (lane == 0) {
        const int myPoint = blockIdx.y * points + blockIdx.x;
        out_partial_density[myPoint] = rho_val;
      }
    }
  }
}

#endif  // G2G_KERNELS_ENERGY_H
