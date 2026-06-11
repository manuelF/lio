/* CPU-only build stubs for the GPU-resident int3lu Coulomb-fit GEMVs
 * (see cuda/coulomb_fit.cu). ensure() reports ok=0 so int3lu always takes
 * its CPU BLAS path. */
#ifndef GPU_KERNELS
extern "C" {
void int3lu_gpu_invalidate_(void) {}
void int3lu_gpu_finalize_(void) {}
void int3lu_gpu_prefetch_(double*, float*, int*, int*, int*) {}
void int3lu_gpu_ensure_(double*, float*, int*, int*, int*, int* ok) {
  *ok = 0;
}
void int3lu_gpu_rc_(double*, float*, double*, float*) {}
void int3lu_gpu_terms_(double*, float*, double*, float*) {}
}
#endif
