// -*- mode: c -*-

/**
 * @file energy.h
 * @brief CUDA kernels for numerical integration of the density and its
 * derivatives.
 *
 * This file implements the evaluation of the electron density rho(r) and its
 * spatial derivatives (gradients and Hessians) on a numerical grid for
 * closed-shell systems. It utilizes texture memory for efficient access to the
 * density matrix (RMM) and shared memory to minimize global memory bandwidth
 * for basis function values.
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
 * REDUCTION HELPERS
 * ==========================================================================================
 */

/**
 * @brief Performs a warp-level reduction for a scalar value.
 */
template <typename T>
__device__ __forceinline__ void warpReduceScalar(volatile T* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

/**
 * @brief Performs a warp-level reduction for a 3D vector.
 */
template <typename T>
__device__ __forceinline__ void warpReduceVector3(
    volatile vec_type<T, 3>* sdata, int tid) {
  sdata[tid].x += sdata[tid + 32].x;
  sdata[tid].y += sdata[tid + 32].y;
  sdata[tid].z += sdata[tid + 32].z;
  sdata[tid].x += sdata[tid + 16].x;
  sdata[tid].y += sdata[tid + 16].y;
  sdata[tid].z += sdata[tid + 16].z;
  sdata[tid].x += sdata[tid + 8].x;
  sdata[tid].y += sdata[tid + 8].y;
  sdata[tid].z += sdata[tid + 8].z;
  sdata[tid].x += sdata[tid + 4].x;
  sdata[tid].y += sdata[tid + 4].y;
  sdata[tid].z += sdata[tid + 4].z;
  sdata[tid].x += sdata[tid + 2].x;
  sdata[tid].y += sdata[tid + 2].y;
  sdata[tid].z += sdata[tid + 2].z;
  sdata[tid].x += sdata[tid + 1].x;
  sdata[tid].y += sdata[tid + 1].y;
  sdata[tid].z += sdata[tid + 1].z;
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
 * @tparam scalar_type    Precision type.
 * @tparam lda           If true, only compute density. If false, compute
 * gradients/Hessians.
 */
template <class scalar_type, bool lda>
__global__ void gpu_compute_density(
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

  for (int bj = 0; bj <= min_i; bj += DENSITY_BLOCK_SIZE) {
    __syncthreads();
    if (bj + tid < m) {
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
      bool full_block = (i >= bj + DENSITY_BLOCK_SIZE - 1);
#pragma unroll 4
      for (int j = 0; j < DENSITY_BLOCK_SIZE; j++) {
        if (full_block || (bj + j) <= i) {
          scalar_type rdm = fetch(rmm_input_gpu_tex, (float)(bj + j), (float)i);
          scalar_type fj_val = fj_sh[j];
          w += rdm * fj_val;
          if (!lda) {
            w3 += fgj_sh[j] * rdm;
            ww1 += fh1j_sh[j] * rdm;
            ww2 += fh2j_sh[j] * rdm;
          }
        }
        if (valid_thread2 && (full_block || (bj + j) <= i2)) {
          scalar_type rdm2 =
              fetch(rmm_input_gpu_tex, (float)(bj + j), (float)i2);
          scalar_type fj_val = fj_sh[j];
          w2 += rdm2 * fj_val;
          if (!lda) {
            w32 += fgj_sh[j] * rdm2;
            ww12 += fh1j_sh[j] * rdm2;
            ww22 += fh2j_sh[j] * rdm2;
          }
        }
      }
    }
  }

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

  __syncthreads();
  fj_sh[tid] = partial_rho;
  fgj_sh[tid] = dxyz;
  fh1j_sh[tid] = dd1;
  fh2j_sh[tid] = dd2;
  __syncthreads();

  if (tid < 32) {
    warpReduceScalar(fj_sh, tid);
    if (!lda) {
      warpReduceVector3(fgj_sh, tid);
      warpReduceVector3(fh1j_sh, tid);
      warpReduceVector3(fh2j_sh, tid);
    }
  }

  if (tid == 0) {
    const int myPoint = blockIdx.y * points + blockIdx.x;
    out_partial_density[myPoint] = fj_sh[0];
    out_dxyz[myPoint] = vec_type<scalar_type, 4>(fgj_sh[0]);
    out_dd1[myPoint] = vec_type<scalar_type, 4>(fh1j_sh[0]);
    out_dd2[myPoint] = vec_type<scalar_type, 4>(fh2j_sh[0]);
  }
}

#endif  // G2G_KERNELS_ENERGY_H