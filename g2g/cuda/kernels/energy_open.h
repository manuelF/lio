#if FULL_DOUBLE
/*
static __inline__ __device__ double fetch_double(cudaTextureObject_t t, float x,
float y)
{
   int2 v = tex2D<int2>(t,x,y);
   return __hiloint2double(v.y, v.x);
}*/
#define fetch(t, x, y) fetch_double(t, x, y)
#else
#define fetch(t, x, y) tex2D<float>(t, x, y)
#endif

// As with gpu_compute_density, point_weights and the compute_* template flags
// were never referenced in the body — only out_partial_density_*/dxyz_*/dd*
// are written. Collapses 3 identical specializations into 1.
//
// `single_pointer=true` is a host-side hint that block_height==1 (i.e.
// group_m <= 2*DENSITY_BLOCK_SIZE), in which case the second basis-row
// pointer (i2 = i + DENSITY_BLOCK_SIZE) is guaranteed to fall past m for
// every thread (valid_thread2 == false) and the entire i2 accumulation
// branch is dead code. Templating on this lets the compiler eliminate the
// dead branch's registers (w2_*, w32_*, ww12_*, ww22_*, Fi2/Fgi2/Fhi*2),
// dropping observed reg-pressure from 80 → ~50 and lifting theoretical
// occupancy on Ampere from 50% → ~75%. Bit-exact: the i2 branch was
// already runtime-dead at block_height==1, the template just lets ptxas
// see it.
template <class scalar_type, bool lda, bool single_pointer>
__global__ void gpu_compute_density_opened(
    cudaTextureObject_t rmm_input_gpu_tex,
    cudaTextureObject_t rmm_input_gpu_tex2, uint points,
    const scalar_type* __restrict__ function_values,
    const vec_type<scalar_type, 4>* __restrict__ gradient_values,
    const vec_type<scalar_type, 4>* __restrict__ hessian_values, uint m,
    scalar_type* __restrict__ out_partial_density_a,
    vec_type<scalar_type, 4>* __restrict__ out_dxyz_a,
    vec_type<scalar_type, 4>* __restrict__ out_dd1_a,
    vec_type<scalar_type, 4>* __restrict__ out_dd2_a,
    scalar_type* __restrict__ out_partial_density_b,
    vec_type<scalar_type, 4>* __restrict__ out_dxyz_b,
    vec_type<scalar_type, 4>* __restrict__ out_dd1_b,
    vec_type<scalar_type, 4>* __restrict__ out_dd2_b) {
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
  bool valid_thread2 = single_pointer ? false : (i2 < m);

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
    if (!single_pointer) {
      w32_a = ww12_a = ww22_a = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
      w32_b = ww12_b = ww22_b = vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
    }
  }

  int position = threadIdx.x;

  // 4-wide shared tiles: the inner loop is LSU-bound, so each j-iteration
  // must load these with LDS.128 instead of scalar loads (see energy.h).
  __shared__ scalar_type fj_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 4> fgj_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 4> fh1j_sh[DENSITY_BLOCK_SIZE];
  __shared__ vec_type<scalar_type, 4> fh2j_sh[DENSITY_BLOCK_SIZE];

  // Si nos vamos a pasar del bloque con el segundo puntero, hacemos que haga la
  // misma cuenta
  if (min_i > m) {
    min_i = min_i - DENSITY_BLOCK_SIZE;
  }

  for (int bj = 0; bj <= min_i; bj += DENSITY_BLOCK_SIZE) {
    // Density deberia ser GET_DENSITY_BLOCK_SIZE

    __syncthreads();
    if (bj + position < m) {
      fj_sh[position] = function_values[(m)*point + (bj + position)];
      if (!lda) {
        fgj_sh[position] = gradient_values[(m)*point + (bj + position)];
        fh1j_sh[position] =
            hessian_values[(m) * 2 * point + (2 * (bj + position) + 0)];
        fh2j_sh[position] =
            hessian_values[(m) * 2 * point + (2 * (bj + position) + 1)];
      }
    }

    __syncthreads();
    scalar_type fjreg = 0.0f;
    vec_type<scalar_type, 3> fgjreg =
        vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
    vec_type<scalar_type, 3> fh1jreg =
        vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);
    vec_type<scalar_type, 3> fh2jreg =
        vec_type<scalar_type, 3>(0.0f, 0.0f, 0.0f);

    // Tighter loop bound for small groups: the shared fj_sh/fgj_sh/fh1j_sh/
    // fh2j_sh slots past index (m-bj-1) are never populated (the shared store
    // is gated by `bj+position < m`), and the guarded fetches inside this
    // loop short-circuit anyway. For small group_m the original 64-trip loop wastes 
    // 40+ iterations doing dead shared loads + branch evaluation. 
    // Uniform across the block (no divergence). Bit-exact
    // — only skips iterations that were already no-ops.
    int j_max = (int)m - bj;
    if (j_max > DENSITY_BLOCK_SIZE) j_max = DENSITY_BLOCK_SIZE;
    if (valid_thread) {
      for (int j = 0; j < j_max; j++) {
        fjreg = fj_sh[j];
        if (!lda) {
          // Full 4-wide struct copies so the compiler emits LDS.128.
          const vec_type<scalar_type, 4> fgj4 = fgj_sh[j];
          const vec_type<scalar_type, 4> fh1j4 = fh1j_sh[j];
          const vec_type<scalar_type, 4> fh2j4 = fh2j_sh[j];
          fgjreg = vec_type<scalar_type, 3>(fgj4.x, fgj4.y, fgj4.z);
          fh1jreg = vec_type<scalar_type, 3>(fh1j4.x, fh1j4.y, fh1j4.z);
          fh2jreg = vec_type<scalar_type, 3>(fh2j4.x, fh2j4.y, fh2j4.z);
        }

        if ((bj + j) <= i) {
          // Fetch is a  macro for tex2D
          scalar_type rdm_this_thread_a =
              fetch(rmm_input_gpu_tex, (float)(bj + j), (float)i);
          scalar_type rdm_this_thread_b =
              fetch(rmm_input_gpu_tex2, (float)(bj + j), (float)i);

          w_a += rdm_this_thread_a * fjreg;
          w_b += rdm_this_thread_b * fjreg;

          if (!lda) {
            w3_a += fgjreg * rdm_this_thread_a;
            ww1_a += fh1jreg * rdm_this_thread_a;
            ww2_a += fh2jreg * rdm_this_thread_a;

            w3_b += fgjreg * rdm_this_thread_b;
            ww1_b += fh1jreg * rdm_this_thread_b;
            ww2_b += fh2jreg * rdm_this_thread_b;
          }
        }

        if (!single_pointer && valid_thread2 && ((bj + j) <= i2)) {
          scalar_type rdm_this_thread2_a =
              fetch(rmm_input_gpu_tex, (float)(bj + j), (float)i2);
          scalar_type rdm_this_thread2_b =
              fetch(rmm_input_gpu_tex2, (float)(bj + j), (float)i2);
          w2_a += rdm_this_thread2_a * fjreg;
          w2_b += rdm_this_thread2_b * fjreg;
          if (!lda) {
            w32_a += fgjreg * rdm_this_thread2_a;
            ww12_a += fh1jreg * rdm_this_thread2_a;
            ww22_a += fh2jreg * rdm_this_thread2_a;

            w32_b += fgjreg * rdm_this_thread2_b;
            ww12_b += fh1jreg * rdm_this_thread2_b;
            ww22_b += fh2jreg * rdm_this_thread2_b;
          }
        }
      }
    }
  }
  if (valid_thread) {
    // Loads relocated to after the bj loop (mirrors closed-shell energy.h):
    // these registers are not consumed inside the loop, so holding them live
    // across it was inflating register pressure (96 -> ~56 regs target).
    scalar_type Fi = function_values[(m)*point + i];
    vec_type<scalar_type, 3> Fgi, Fhi1, Fhi2;
    if (!lda) {
      Fgi = vec_type<scalar_type, 3>(gradient_values[(m)*point + i]);
      Fhi1 = vec_type<scalar_type, 3>(
          hessian_values[(m) * 2 * point + (2 * i + 0)]);
      Fhi2 = vec_type<scalar_type, 3>(
          hessian_values[(m) * 2 * point + (2 * i + 1)]);
    }

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
  if (!single_pointer && valid_thread2) {
    scalar_type Fi2 = function_values[(m)*point + i2];
    vec_type<scalar_type, 3> Fgi2, Fhi12, Fhi22;
    if (!lda) {
      Fgi2 = vec_type<scalar_type, 3>(gradient_values[(m)*point + i2]);
      Fhi12 = vec_type<scalar_type, 3>(
          hessian_values[(m) * 2 * point + (2 * i2 + 0)]);
      Fhi22 = vec_type<scalar_type, 3>(
          hessian_values[(m) * 2 * point + (2 * i2 + 1)]);
    }

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

  __syncthreads();
  // Reusing per-block shared memory in order to perform per-block accumulation.
  // Alpha density
  if (valid_thread) {
    fj_sh[position] = partial_density_a;
    fgj_sh[position] = vec_type<scalar_type, 4>(dxyz_a.x, dxyz_a.y, dxyz_a.z,
                                                scalar_type(0.0f));
    fh1j_sh[position] = vec_type<scalar_type, 4>(dd1_a.x, dd1_a.y, dd1_a.z,
                                                 scalar_type(0.0f));
    fh2j_sh[position] = vec_type<scalar_type, 4>(dd2_a.x, dd2_a.y, dd2_a.z,
                                                 scalar_type(0.0f));
  } else {
    fj_sh[position] = scalar_type(0.0f);
    fgj_sh[position] = vec_type<scalar_type, 4>(0.0f, 0.0f, 0.0f, 0.0f);
    fh1j_sh[position] = vec_type<scalar_type, 4>(0.0f, 0.0f, 0.0f, 0.0f);
    fh2j_sh[position] = vec_type<scalar_type, 4>(0.0f, 0.0f, 0.0f, 0.0f);
  }
  __syncthreads();

  // DENSITY_BLOCK_SIZE (64) spans multiple warps, so this tree reduction
  // needs a __syncthreads inside each iteration to be race-free — matches
  // the closed-shell energy.h pattern. (Was missing on the open path.)
  for (int j = 2; j <= DENSITY_BLOCK_SIZE; j = j * 2) {
    int index = position + DENSITY_BLOCK_SIZE / j;
    if (position < DENSITY_BLOCK_SIZE / j) {
      fj_sh[position] += fj_sh[index];
      fgj_sh[position] += fgj_sh[index];
      fh1j_sh[position] += fh1j_sh[index];
      fh2j_sh[position] += fh2j_sh[index];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int myPoint = blockIdx.y * points + blockIdx.x;
    out_partial_density_a[myPoint] = fj_sh[position];
    out_dxyz_a[myPoint] = vec_type<scalar_type, 4>(fgj_sh[position]);
    out_dd1_a[myPoint] = vec_type<scalar_type, 4>(fh1j_sh[position]);
    out_dd2_a[myPoint] = vec_type<scalar_type, 4>(fh2j_sh[position]);
  }

  __syncthreads();
  // Beta density.
  if (valid_thread) {
    fj_sh[position] = partial_density_b;
    fgj_sh[position] = vec_type<scalar_type, 4>(dxyz_b.x, dxyz_b.y, dxyz_b.z,
                                                scalar_type(0.0f));
    fh1j_sh[position] = vec_type<scalar_type, 4>(dd1_b.x, dd1_b.y, dd1_b.z,
                                                 scalar_type(0.0f));
    fh2j_sh[position] = vec_type<scalar_type, 4>(dd2_b.x, dd2_b.y, dd2_b.z,
                                                 scalar_type(0.0f));
  } else {
    fj_sh[position] = scalar_type(0.0f);
    fgj_sh[position] = vec_type<scalar_type, 4>(0.0f, 0.0f, 0.0f, 0.0f);
    fh1j_sh[position] = vec_type<scalar_type, 4>(0.0f, 0.0f, 0.0f, 0.0f);
    fh2j_sh[position] = vec_type<scalar_type, 4>(0.0f, 0.0f, 0.0f, 0.0f);
  }
  __syncthreads();

  for (int j = 2; j <= DENSITY_BLOCK_SIZE; j = j * 2) {
    int index = position + DENSITY_BLOCK_SIZE / j;
    if (position < DENSITY_BLOCK_SIZE / j) {
      fj_sh[position] += fj_sh[index];
      fgj_sh[position] += fgj_sh[index];
      fh1j_sh[position] += fh1j_sh[index];
      fh2j_sh[position] += fh2j_sh[index];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int myPoint = blockIdx.y * points + blockIdx.x;
    out_partial_density_b[myPoint] = fj_sh[position];
    out_dxyz_b[myPoint] = vec_type<scalar_type, 4>(fgj_sh[position]);
    out_dd1_b[myPoint] = vec_type<scalar_type, 4>(fh1j_sh[position]);
    out_dd2_b[myPoint] = vec_type<scalar_type, 4>(fh2j_sh[position]);
  }
}
