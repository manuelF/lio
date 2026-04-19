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

  // --- Two-warp reduction (numerically equivalent to the original volatile pattern) ---
  //
  // Block = 64 threads = 2 warps.  The original volatile code paired each thread
  // in warp 0 with the corresponding thread in warp 1 (sdata[tid] += sdata[tid+32])
  // BEFORE the intra-warp binary tree.  To replicate that floating-point order:
  //
  // Step 1: all 64 threads write their partials to shared memory (arrays are
  //         already DENSITY_BLOCK_SIZE=64 long for the bj-loop cache).
  //         One __syncthreads() makes them visible.
  //
  // Step 2: warp 0 loads val[lane] = fj_sh[lane] + fj_sh[lane+32]
  //         (cross-warp pairing — matches old first step), then reduces with
  //         __shfl_down_sync (register-only from here).
  //
  // Alpha and beta are done with two separate shared-memory writes, each
  // needing one __syncthreads(), for 2 syncs total (same as before).

  int lane = position & 31;
  int warp = position >> 5;  // 0 or 1

  // Barrier: ensure all threads finish reads of fj_sh[j]/fgj_sh[j]/fh1j_sh[j]/
  // fh2j_sh[j] from the outer bj loop before we reuse that shared memory for
  // reduction partials. Without this, threads that finished the inner j-loop
  // early (e.g. !valid_thread branches, or early-exit bounds) can race with
  // threads still reading the cached basis values. FULL_DOUBLE exposes the
  // race because 64-bit shared-memory writes can tear into partial 32-bit
  // reads — the closed-shell kernel had the same bug (energy.h).
  __syncthreads();

  // Step 1a: store alpha partials, sync, warp-0 cross-warp pair + shuffle.
  // A second __syncthreads() after the alpha block (below) ensures warp 0
  // finishes reading fj_sh[32..63] before warp 1 overwrites them with beta.
  fj_sh[position] = partial_density_a;
  if (!lda) {
    fgj_sh[position]  = dxyz_a;
    fh1j_sh[position] = dd1_a;
    fh2j_sh[position] = dd2_a;
  }
  __syncthreads();

  if (warp == 0) {
    scalar_type rho_a = fj_sh[lane] + fj_sh[lane + 32];
    rho_a = warpReduceScalar(rho_a);
    if (!lda) {
      vec_type<scalar_type, 3> dxyz_av = fgj_sh[lane]  + fgj_sh[lane + 32];
      vec_type<scalar_type, 3> dd1_av  = fh1j_sh[lane] + fh1j_sh[lane + 32];
      vec_type<scalar_type, 3> dd2_av  = fh2j_sh[lane] + fh2j_sh[lane + 32];
      dxyz_av = warpReduceVector3(dxyz_av);
      dd1_av  = warpReduceVector3(dd1_av);
      dd2_av  = warpReduceVector3(dd2_av);
      if (lane == 0) {
        const int myPoint = blockIdx.y * points + blockIdx.x;
        out_partial_density_a[myPoint] = rho_a;
        out_dxyz_a[myPoint] = vec_type<scalar_type, 4>(dxyz_av);
        out_dd1_a[myPoint]  = vec_type<scalar_type, 4>(dd1_av);
        out_dd2_a[myPoint]  = vec_type<scalar_type, 4>(dd2_av);
      }
    } else {
      if (lane == 0) {
        const int myPoint = blockIdx.y * points + blockIdx.x;
        out_partial_density_a[myPoint] = rho_a;
      }
    }
  }

  // Barrier: prevent warp 1 from writing beta data into fj_sh[32..63] before
  // warp 0 has finished reading those slots for the alpha reduction above.
  __syncthreads();

  // Step 1b: store beta partials, sync, warp-0 cross-warp pair + shuffle
  fj_sh[position] = partial_density_b;
  if (!lda) {
    fgj_sh[position]  = dxyz_b;
    fh1j_sh[position] = dd1_b;
    fh2j_sh[position] = dd2_b;
  }
  __syncthreads();

  if (warp == 0) {
    scalar_type rho_b = fj_sh[lane] + fj_sh[lane + 32];
    rho_b = warpReduceScalar(rho_b);
    if (!lda) {
      vec_type<scalar_type, 3> dxyz_bv = fgj_sh[lane]  + fgj_sh[lane + 32];
      vec_type<scalar_type, 3> dd1_bv  = fh1j_sh[lane] + fh1j_sh[lane + 32];
      vec_type<scalar_type, 3> dd2_bv  = fh2j_sh[lane] + fh2j_sh[lane + 32];
      dxyz_bv = warpReduceVector3(dxyz_bv);
      dd1_bv  = warpReduceVector3(dd1_bv);
      dd2_bv  = warpReduceVector3(dd2_bv);
      if (lane == 0) {
        const int myPoint = blockIdx.y * points + blockIdx.x;
        out_partial_density_b[myPoint] = rho_b;
        out_dxyz_b[myPoint] = vec_type<scalar_type, 4>(dxyz_bv);
        out_dd1_b[myPoint]  = vec_type<scalar_type, 4>(dd1_bv);
        out_dd2_b[myPoint]  = vec_type<scalar_type, 4>(dd2_bv);
      }
    } else {
      if (lane == 0) {
        const int myPoint = blockIdx.y * points + blockIdx.x;
        out_partial_density_b[myPoint] = rho_b;
      }
    }
  }
}
