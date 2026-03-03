#if FULL_DOUBLE
#define fetch(t, x, y) fetch_double(t, x, y)
#else
#define fetch(t, x, y) tex2D<float>(t, x, y)
#endif

/**
 * @brief CUDA kernel to compute density and derivatives for open-shell
 * (alpha + beta spin) systems.
 *
 * Block: dim3(DENSITY_BLOCK_SIZE, 1, 1) = 64 threads = 2 warps.
 * Grid:  dim3(npoints, block_height, 1).
 *
 * Reduction: same two-warp shuffle strategy as energy.h, applied twice
 * (once for alpha spin, once for beta spin). All shuffle work is register-only;
 * shared memory is only touched for the 2-slot cross-warp communication.
 * Reduces from 4 __syncthreads() to 2 versus the volatile pattern.
 */
template <class scalar_type, bool compute_energy, bool compute_factor, bool lda>
__global__ void gpu_compute_density_opened(
    cudaTextureObject_t rmm_input_gpu_tex,
    cudaTextureObject_t rmm_input_gpu_tex2, const scalar_type* point_weights,
    uint points, const scalar_type* function_values,
    const vec_type<scalar_type, 4>* gradient_values,
    const vec_type<scalar_type, 4>* hessian_values, uint m,
    scalar_type* out_partial_density_a, vec_type<scalar_type, 4>* out_dxyz_a,
    vec_type<scalar_type, 4>* out_dd1_a, vec_type<scalar_type, 4>* out_dd2_a,
    scalar_type* out_partial_density_b, vec_type<scalar_type, 4>* out_dxyz_b,
    vec_type<scalar_type, 4>* out_dd1_b, vec_type<scalar_type, 4>* out_dd2_b) {
  uint point = blockIdx.x;
  uint i = threadIdx.x + blockIdx.y * 2 * DENSITY_BLOCK_SIZE;
  uint i2 = i + DENSITY_BLOCK_SIZE;
  uint min_i = blockIdx.y * 2 * DENSITY_BLOCK_SIZE + DENSITY_BLOCK_SIZE;

  scalar_type partial_density_a(0.0f);
  scalar_type partial_density_b(0.0f);

  vec_type<scalar_type, 3> dxyz_a, dd1_a, dd2_a;
  vec_type<scalar_type, 3> dxyz_b, dd1_b, dd2_b;
  dxyz_a = dd1_a = dd2_a = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
  dxyz_b = dd1_b = dd2_b = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);

  bool valid_thread = (i < m);
  bool valid_thread2 = (i2 < m);

  scalar_type w_a = 0.0f;
  scalar_type w_b = 0.0f;
  scalar_type w2_a = 0.0f;
  scalar_type w2_b = 0.0f;
  vec_type<scalar_type, 3> w3_a, ww1_a, ww2_a;
  vec_type<scalar_type, 3> w3_b, ww1_b, ww2_b;
  vec_type<scalar_type, 3> w32_a, ww12_a, ww22_a;
  vec_type<scalar_type, 3> w32_b, ww12_b, ww22_b;

  if (!lda) {
    w3_a = ww1_a = ww2_a = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
    w3_b = ww1_b = ww2_b = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
    w32_a = ww12_a = ww22_a = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
    w32_b = ww12_b = ww22_b = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
  }

  scalar_type Fi, Fi2;
  vec_type<scalar_type, 3> Fgi, Fhi1, Fhi2, Fgi2, Fhi12, Fhi22;

  int position = threadIdx.x;

  // Shared memory for function-value caching (bj-loop) and 2-slot cross-warp
  // communication after the shuffle reduction.
  __shared__ scalar_type fj_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 3> fgj_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 3> fh1j_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 3> fh2j_sh[DENSITY_BLOCK_SIZE];

  if (min_i > m) {
    min_i = min_i - DENSITY_BLOCK_SIZE;
  }

  if (valid_thread2) {
    Fi2 = function_values[(m)*point + i2];
    if (!lda) {
      Fgi2 = vec_type<scalar_type, 3>(gradient_values[(m)*point + i2]);
      Fhi12 = vec_type<scalar_type, 3>(
          hessian_values[(m) * 2 * point + (2 * i2 + 0)]);
      Fhi22 = vec_type<scalar_type, 3>(
          hessian_values[(m) * 2 * point + (2 * i2 + 1)]);
    }
  }
  if (valid_thread) {
    Fi = function_values[(m)*point + i];
    if (!lda) {
      Fgi = vec_type<scalar_type, 3>(gradient_values[(m)*point + i]);
      Fhi1 = vec_type<scalar_type, 3>(
          hessian_values[(m) * 2 * point + (2 * i + 0)]);
      Fhi2 = vec_type<scalar_type, 3>(
          hessian_values[(m) * 2 * point + (2 * i + 1)]);
    }
  }

  for (int bj = 0; bj <= (int)min_i; bj += DENSITY_BLOCK_SIZE) {
    __syncthreads();
    if (bj + position < (int)m) {
      fj_sh[position] = function_values[(m)*point + (bj + position)];
      if (!lda) {
        fgj_sh[position] = vec_type<scalar_type, 3>(
            gradient_values[(m)*point + (bj + position)]);
        fh1j_sh[position] = vec_type<scalar_type, 3>(
            hessian_values[(m) * 2 * point + (2 * (bj + position) + 0)]);
        fh2j_sh[position] = vec_type<scalar_type, 3>(
            hessian_values[(m) * 2 * point + (2 * (bj + position) + 1)]);
      }
    }

    __syncthreads();

    if (valid_thread) {
      bool full_block = (i >= (uint)(bj + DENSITY_BLOCK_SIZE - 1));
#pragma unroll 4
      for (int j = 0; j < DENSITY_BLOCK_SIZE; j++) {
        if (full_block || (bj + j) <= (int)i) {
          scalar_type rdm_a =
              fetch(rmm_input_gpu_tex, (float)(bj + j), (float)i);
          scalar_type rdm_b =
              fetch(rmm_input_gpu_tex2, (float)(bj + j), (float)i);
          scalar_type fj_val = fj_sh[j];
          w_a += rdm_a * fj_val;
          w_b += rdm_b * fj_val;
          if (!lda) {
            w3_a  += fgj_sh[j]  * rdm_a;
            ww1_a += fh1j_sh[j] * rdm_a;
            ww2_a += fh2j_sh[j] * rdm_a;
            w3_b  += fgj_sh[j]  * rdm_b;
            ww1_b += fh1j_sh[j] * rdm_b;
            ww2_b += fh2j_sh[j] * rdm_b;
          }
        }

        if (valid_thread2 && (full_block || (bj + j) <= (int)i2)) {
          scalar_type rdm2_a =
              fetch(rmm_input_gpu_tex, (float)(bj + j), (float)i2);
          scalar_type rdm2_b =
              fetch(rmm_input_gpu_tex2, (float)(bj + j), (float)i2);
          scalar_type fj_val = fj_sh[j];
          w2_a += rdm2_a * fj_val;
          w2_b += rdm2_b * fj_val;
          if (!lda) {
            w32_a  += fgj_sh[j]  * rdm2_a;
            ww12_a += fh1j_sh[j] * rdm2_a;
            ww22_a += fh2j_sh[j] * rdm2_a;
            w32_b  += fgj_sh[j]  * rdm2_b;
            ww12_b += fh1j_sh[j] * rdm2_b;
            ww22_b += fh2j_sh[j] * rdm2_b;
          }
        }
      }
    }
  }

  // --- Per-thread local partial results ---
  if (valid_thread) {
    partial_density_a = Fi * w_a;
    partial_density_b = Fi * w_b;
    if (!lda) {
      dxyz_a = Fgi * w_a + w3_a * Fi;
      dd1_a = Fgi * w3_a * 2.0f + Fhi1 * w_a + ww1_a * Fi;

      dxyz_b = Fgi * w_b + w3_b * Fi;
      dd1_b = Fgi * w3_b * 2.0f + Fhi1 * w_b + ww1_b * Fi;

      vec_type<scalar_type, 3> FgXXY(Fgi.x, Fgi.x, Fgi.y);
      vec_type<scalar_type, 3> w3YZZ_a(w3_a.y, w3_a.z, w3_a.z);
      vec_type<scalar_type, 3> w3YZZ_b(w3_b.y, w3_b.z, w3_b.z);
      vec_type<scalar_type, 3> FgiYZZ(Fgi.y, Fgi.z, Fgi.z);
      vec_type<scalar_type, 3> w3XXY_a(w3_a.x, w3_a.x, w3_a.y);
      vec_type<scalar_type, 3> w3XXY_b(w3_b.x, w3_b.x, w3_b.y);

      dd2_a = FgXXY * w3YZZ_a + FgiYZZ * w3XXY_a + Fhi2 * w_a + ww2_a * Fi;
      dd2_b = FgXXY * w3YZZ_b + FgiYZZ * w3XXY_b + Fhi2 * w_b + ww2_b * Fi;
    }
  }
  if (valid_thread2) {
    partial_density_a += Fi2 * w2_a;
    partial_density_b += Fi2 * w2_b;
    if (!lda) {
      dxyz_a += Fgi2 * w2_a + w32_a * Fi2;
      dd1_a += Fgi2 * w32_a * 2.0f + Fhi12 * w2_a + ww12_a * Fi2;
      dxyz_b += Fgi2 * w2_b + w32_b * Fi2;
      dd1_b += Fgi2 * w32_b * 2.0f + Fhi12 * w2_b + ww12_b * Fi2;

      vec_type<scalar_type, 3> FgXXY(Fgi2.x, Fgi2.x, Fgi2.y);
      vec_type<scalar_type, 3> w3YZZ_a(w32_a.y, w32_a.z, w32_a.z);
      vec_type<scalar_type, 3> w3YZZ_b(w32_b.y, w32_b.z, w32_b.z);
      vec_type<scalar_type, 3> FgiYZZ(Fgi2.y, Fgi2.z, Fgi2.z);
      vec_type<scalar_type, 3> w3XXY_a(w32_a.x, w32_a.x, w32_a.y);
      vec_type<scalar_type, 3> w3XXY_b(w32_b.x, w32_b.x, w32_b.y);

      dd2_a += FgXXY * w3YZZ_a + FgiYZZ * w3XXY_a + Fhi22 * w2_a + ww22_a * Fi2;
      dd2_b += FgXXY * w3YZZ_b + FgiYZZ * w3XXY_b + Fhi22 * w2_b + ww22_b * Fi2;
    }
  }

  // --- Two-warp reduction via __shfl_down_sync ---
  //
  // All shuffle work is pure register — no shared memory, no store fences.
  // Shared memory (fj_sh[0..1]) is used only for the 2-slot cross-warp
  // communication after the intra-warp reductions.
  //
  // Alpha and beta are reduced sequentially, sharing the same shared arrays.
  // This halves the number of __syncthreads() calls versus the volatile pattern
  // (2 syncs instead of 4).

  int lane = position & 31;
  int warp = position >> 5;  // 0 or 1

  // Step 1: shuffle reduce ALL partial sums (register-only, both spins at once)
  partial_density_a = warpReduceScalar(partial_density_a);
  partial_density_b = warpReduceScalar(partial_density_b);
  if (!lda) {
    dxyz_a = warpReduceVector3(dxyz_a);
    dd1_a  = warpReduceVector3(dd1_a);
    dd2_a  = warpReduceVector3(dd2_a);
    dxyz_b = warpReduceVector3(dxyz_b);
    dd1_b  = warpReduceVector3(dd1_b);
    dd2_b  = warpReduceVector3(dd2_b);
  }

  // Step 2: alpha cross-warp accumulation
  if (lane == 0) {
    fj_sh[warp] = partial_density_a;
    if (!lda) {
      fgj_sh[warp]  = dxyz_a;
      fh1j_sh[warp] = dd1_a;
      fh2j_sh[warp] = dd2_a;
    }
  }
  __syncthreads();

  if (position == 0) {
    const int myPoint = blockIdx.y * points + blockIdx.x;
    out_partial_density_a[myPoint] = fj_sh[0] + fj_sh[1];
    if (!lda) {
      out_dxyz_a[myPoint] = vec_type<scalar_type, 4>(fgj_sh[0]  + fgj_sh[1]);
      out_dd1_a[myPoint]  = vec_type<scalar_type, 4>(fh1j_sh[0] + fh1j_sh[1]);
      out_dd2_a[myPoint]  = vec_type<scalar_type, 4>(fh2j_sh[0] + fh2j_sh[1]);
    }
  }

  // Step 3: beta cross-warp accumulation (reuse same shared arrays)
  if (lane == 0) {
    fj_sh[warp] = partial_density_b;
    if (!lda) {
      fgj_sh[warp]  = dxyz_b;
      fh1j_sh[warp] = dd1_b;
      fh2j_sh[warp] = dd2_b;
    }
  }
  __syncthreads();

  if (position == 0) {
    const int myPoint = blockIdx.y * points + blockIdx.x;
    out_partial_density_b[myPoint] = fj_sh[0] + fj_sh[1];
    if (!lda) {
      out_dxyz_b[myPoint] = vec_type<scalar_type, 4>(fgj_sh[0]  + fgj_sh[1]);
      out_dd1_b[myPoint]  = vec_type<scalar_type, 4>(fh1j_sh[0] + fh1j_sh[1]);
      out_dd2_b[myPoint]  = vec_type<scalar_type, 4>(fh2j_sh[0] + fh2j_sh[1]);
    }
  }
}
