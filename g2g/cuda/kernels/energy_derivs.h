// TODO: si se juntara con energy.h (teniendo un if templatizado tipo do_forces,
// que hace que solo se desde donde se debe (j >= i)) se leeria RMM una sola vez

template <class scalar_type>
__global__ void gpu_compute_density_derivs(
    cudaTextureObject_t rmm_input_gpu_tex,
    const scalar_type* __restrict__ function_values,
    const vec_type<scalar_type, 4>* __restrict__ gradient_values,
    const uint* __restrict__ nuc,
    vec_type<scalar_type, 4>* __restrict__ density_deriv, uint points, uint m,
    uint nuc_count) {
  uint point = index_x(blockDim, blockIdx, threadIdx);
  bool valid_thread = (point < points);

  __shared__ scalar_type rdm_sh[DENSITY_DERIV_BATCH_SIZE];
  __shared__ uint nuc_sh[DENSITY_DERIV_BATCH_SIZE2];

  for (uint bi = 0; bi < m; bi += DENSITY_DERIV_BATCH_SIZE2) {
    __syncthreads();
    if (threadIdx.x < DENSITY_DERIV_BATCH_SIZE2) {
      if (bi + threadIdx.x < m) nuc_sh[threadIdx.x] = nuc[bi + threadIdx.x];
    }
    __syncthreads();

    for (uint i = 0; i < DENSITY_DERIV_BATCH_SIZE2 && (i + bi) < m; i++) {
      scalar_type w = 0.0f;

      for (uint bj = 0; bj < m; bj += DENSITY_DERIV_BATCH_SIZE) {
        __syncthreads();
        if (threadIdx.x < DENSITY_DERIV_BATCH_SIZE) {
          // fetch es una macro para tex2D definida en energy.h
          if (bj + threadIdx.x < m)
            rdm_sh[threadIdx.x] = fetch(rmm_input_gpu_tex, (float)(bi + i),
                                        (float)(bj + threadIdx.x));
          else
            rdm_sh[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        if (valid_thread) {
          for (uint j = 0; j < DENSITY_DERIV_BATCH_SIZE && (bj + j) < m; j++) {
            scalar_type fj =
                function_values[COALESCED_DIMENSION(points) * (bj + j) + point];
            w += rdm_sh[j] * fj * ((bi + i) == (bj + j) ? 2 : 1);
          }
        }
      }

      if (valid_thread) {
        vec_type<scalar_type, 4> Fgi =
            gradient_values[COALESCED_DIMENSION(points) * (bi + i) + point];
        uint nuci = nuc_sh[i];
        density_deriv[COALESCED_DIMENSION(points) * nuci + point] -= Fgi * w;
        // TODO: esto accede demasiado a memoria, quizas se pueda acumular en
        // sh_mem
      }
    }
  }
}

//===================================================================================================================
// GEMM-path helpers for the closed-shell forces gradient on large groups.
//
// The weighting matvec  W[i][point] = sum_j RDM''[i][j]*f_j[point]  is computed
// as a single dense cuBLAS GEMM  W = F * RDM''  (full-GPU tiled, reads F once),
// then the light kernel below applies the gradient and scatters per nucleus.

// Build a dense column-major (m x m) copy of the gathered symmetric local
// density matrix with its diagonal doubled (the "(i==j ? 2 : 1)" factor).
// rdm_local is the pitch2D gather buffer: element RDM[a][b] lives at
// rdm_local[b*ld + a] (ld = rmm_width). The cuBLAS GEMM consumes rdmpp as the
// (K x N) right operand with leading dimension m, so rdmpp[j + m*i] = RDM''[j][i].
// RDM is symmetric, so the row/col convention is immaterial except on the
// (doubled) diagonal.
template <class scalar_type>
__global__ void gpu_build_rdm_doubled_diag(
    const scalar_type* __restrict__ rdm_local, scalar_type* __restrict__ rdmpp,
    uint m, uint ld) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;  // GEMM column (basis fn i)
  uint j = blockIdx.y * blockDim.y + threadIdx.y;  // GEMM row    (basis fn j)
  if (i >= m || j >= m) return;
  scalar_type v = rdm_local[i * ld + j];  // RDM[j][i]
  if (i == j) v *= (scalar_type)2.0;
  rdmpp[j + m * i] = v;
}

// Fused gradient-apply + point-reduction + per-nucleus scatter, consuming the
// pre-weighted W matrix (column-major, same layout as function_values: element
// W[i][point] at wmat[ld*i + point], ld = COALESCED_DIMENSION(points)).
//
// One block per basis function i reduces over points with coalesced reads:
//   G_i         = sum_point factor[point] * W[i][point] * grad_i[point]   (vec3)
//   forces[nuc] = -sum_{i : nuc_i = nuc} G_i
// then thread 0 atomic-accumulates -G_i into forces[nuc_i]. Launch with a
// power-of-two block size (FORCE_BLOCK_SIZE).
template <class scalar_type>
__global__ void gpu_forces_fused_gemm(
    const scalar_type* __restrict__ wmat,
    const vec_type<scalar_type, 4>* __restrict__ gradient_values,
    const scalar_type* __restrict__ factors, const uint* __restrict__ nuc,
    vec_type<scalar_type, 4>* __restrict__ forces, uint points, uint m) {
  const uint i = blockIdx.x;
  if (i >= m) return;
  const uint ld = COALESCED_DIMENSION(points);
  const scalar_type* __restrict__ wrow = wmat + (size_t)ld * i;
  const vec_type<scalar_type, 4>* __restrict__ grow =
      gradient_values + (size_t)ld * i;

  scalar_type sx = 0, sy = 0, sz = 0;
  for (uint p = threadIdx.x; p < points; p += blockDim.x) {
    scalar_type fw = factors[p] * wrow[p];
    vec_type<scalar_type, 4> g = grow[p];
    sx += fw * g.x;
    sy += fw * g.y;
    sz += fw * g.z;
  }

  __shared__ scalar_type rx[FORCE_BLOCK_SIZE];
  __shared__ scalar_type ry[FORCE_BLOCK_SIZE];
  __shared__ scalar_type rz[FORCE_BLOCK_SIZE];
  rx[threadIdx.x] = sx;
  ry[threadIdx.x] = sy;
  rz[threadIdx.x] = sz;
  __syncthreads();
  for (uint s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      rx[threadIdx.x] += rx[threadIdx.x + s];
      ry[threadIdx.x] += ry[threadIdx.x + s];
      rz[threadIdx.x] += rz[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    uint nuci = nuc[i];
    atomicAdd(&forces[nuci].x, -rx[0]);
    atomicAdd(&forces[nuci].y, -ry[0]);
    atomicAdd(&forces[nuci].z, -rz[0]);
  }
}

//===================================================================================================================
template <class scalar_type>
__global__ void gpu_compute_density_derivs_open(
    cudaTextureObject_t rmm_input_gpu_tex,
    cudaTextureObject_t rmm_input_gpu_tex2, scalar_type* function_values,
    vec_type<scalar_type, 4>* gradient_values, uint* nuc,
    vec_type<scalar_type, 4>* density_deriv_a,
    vec_type<scalar_type, 4>* density_deriv_b, uint points, uint m,
    uint nuc_count) {
  uint point = index_x(blockDim, blockIdx, threadIdx);
  bool valid_thread = (point < points);

  __shared__ scalar_type rdm_a_sh[DENSITY_DERIV_BATCH_SIZE];
  __shared__ scalar_type rdm_b_sh[DENSITY_DERIV_BATCH_SIZE];
  __shared__ uint nuc_sh[DENSITY_DERIV_BATCH_SIZE2];

  for (uint bi = 0; bi < m; bi += DENSITY_DERIV_BATCH_SIZE2) {
    __syncthreads();
    if (threadIdx.x < DENSITY_DERIV_BATCH_SIZE2) {
      if (bi + threadIdx.x < m) nuc_sh[threadIdx.x] = nuc[bi + threadIdx.x];
    }
    __syncthreads();

    for (uint i = 0; i < DENSITY_DERIV_BATCH_SIZE2 && (i + bi) < m; i++) {
      vec_type<scalar_type, 4> Fgi;
      if (valid_thread)
        Fgi = gradient_values[COALESCED_DIMENSION(points) * (bi + i) + point];
      scalar_type w_a = 0.0f;
      scalar_type w_b = 0.0f;

      for (uint bj = 0; bj < m; bj += DENSITY_DERIV_BATCH_SIZE) {
        __syncthreads();
        if (threadIdx.x < DENSITY_DERIV_BATCH_SIZE) {
          // fetch es una macro para tex2D definida en energy.h
          // scalar_type rmd_local = fetch(rmm_input_gpu_tex, (float)(bi+i),
          // (float)(bj+threadIdx.x));
          if (bj + threadIdx.x < m) {
            rdm_a_sh[threadIdx.x] = fetch(rmm_input_gpu_tex, (float)(bi + i),
                                          (float)(bj + threadIdx.x));
            rdm_b_sh[threadIdx.x] = fetch(rmm_input_gpu_tex2, (float)(bi + i),
                                          (float)(bj + threadIdx.x));
          }  // rdm[COALESCED_DIMENSION(m) * (bi + i) + (bj + threadIdx.x)];
          else {
            rdm_a_sh[threadIdx.x] = 0.0f;
            rdm_b_sh[threadIdx.x] = 0.0f;
          }
        }
        __syncthreads();

        if (valid_thread) {
          for (uint j = 0; j < DENSITY_DERIV_BATCH_SIZE && (bj + j) < m; j++) {
            scalar_type fj =
                function_values[COALESCED_DIMENSION(points) * (bj + j) + point];
            w_a += rdm_a_sh[j] * fj * ((bi + i) == (bj + j) ? 2 : 1);
            w_b += rdm_b_sh[j] * fj * ((bi + i) == (bj + j) ? 2 : 1);
          }
        }
      }

      if (valid_thread) {
        uint nuci = nuc_sh[i];
        // TODO: esto accede demasiado a memoria, quizas se pueda acumular en
        // sh_mem
        density_deriv_a[COALESCED_DIMENSION(points) * nuci + point] -=
            Fgi * w_a;
        density_deriv_b[COALESCED_DIMENSION(points) * nuci + point] -=
            Fgi * w_b;
      }
    }
  }
}
