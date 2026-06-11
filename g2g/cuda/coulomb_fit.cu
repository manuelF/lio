/* GPU-resident Coulomb-fit GEMVs for int3lu (MEMO path).
 *
 * int3lu streams the constant 3-center integral matrices cool (Md x kknumd,
 * fp64) and cools (Md x kknums, fp32) twice per SCF iteration (GEMV 'N' for
 * the Rc fitting vector, GEMV 'T' for the Coulomb Fock terms). On the CPU
 * this is DRAM-bandwidth-bound and shares bandwidth with the overlapped g2g
 * partition. Here the matrices are uploaded to the GPU once per geometry
 * (rebuilding/deallocating cool invalidates) and the four GEMVs run through
 * cuBLAS on a private stream, so they interleave with the XC kernels instead
 * of fighting the host memory bus. Everything else (gather/scatter, Ginv
 * DSPMV, energy dots) stays on the CPU with unchanged FP semantics; only the
 * GEMV summation order changes.
 *
 * prefetch_ (called from SCF.f90 right after int3mem, while the GPU is
 * still quiet) allocates the device buffers synchronously — so the
 * GlobalMemoryPool/fgm accounting and later allocators see the
 * reservation — and hands the copy to a background thread. Until the
 * worker flips ready=1, ensure() reports the CPU path; SCF iterations
 * transparently switch to the GPU once the copy has landed. The device
 * copy stays resident until int3lu_gpu_invalidate_(), which callers must
 * invoke before deallocating cool/cools.
 *
 * Falls back to the CPU path permanently when the matrices are too small
 * for the PCIe round-trips to pay off (TD steps on small systems) or when
 * they do not fit in free VRAM after the XC cache budget and temporaries
 * headroom are honored (older GPUs / very large systems).
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <atomic>
#include <thread>

#include "../global_memory_pool.h"

namespace {

// Below this total size the CPU streams the matrices faster than the two
// PCIe round-trips + launch latency cost (~0.5 ms). Derived from ~30 GB/s
// host bandwidth: 2 passes over 16 MB ~= 1 ms CPU.
const size_t MIN_GPU_BYTES = 16 * 1024 * 1024;
// Keep a safety margin of free VRAM for the XC workspaces.
const size_t VRAM_MARGIN = 256 * 1024 * 1024;

struct CoulombFitGpu {
  bool active = false;  // a prefetch was started since the last invalidate
  // 0 = setup/upload in flight (CPU path), 1 = GPU resident, -1 = failed
  std::atomic<int> ready{-1};
  // Set by release_resources() so an in-flight multi-GB upload bails out
  // between chunks instead of making SCF teardown wait for it.
  std::atomic<bool> abort_upload{false};
  std::thread worker;
  int Md = 0, kknumd = 0, kknums = 0;
  double* d_cool = nullptr;   // Md x kknumd
  float* d_cools = nullptr;   // Md x kknums
  double* d_vec_d = nullptr;  // kknumd (rho_gathered / terms)
  float* d_vec_s = nullptr;   // kknums
  double* d_md_d = nullptr;   // Md (Rc / af)
  float* d_md_s = nullptr;    // Md
  void* pinned_cool = nullptr;  // host arrays registered with the driver
  void* pinned_cools = nullptr;
  size_t pool_reserved = 0;  // bytes reserved out of the XC cache budget
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
};

CoulombFitGpu state;

void release_resources() {
  state.abort_upload.store(true);
  if (state.worker.joinable()) state.worker.join();
  state.abort_upload.store(false);
  if (state.stream) cudaStreamSynchronize(state.stream);
  if (state.pinned_cool) cudaHostUnregister(state.pinned_cool);
  if (state.pinned_cools) cudaHostUnregister(state.pinned_cools);
  state.pinned_cool = nullptr;
  state.pinned_cools = nullptr;
  if (state.d_cool) cudaFree(state.d_cool);
  if (state.d_cools) cudaFree(state.d_cools);
  if (state.d_vec_d) cudaFree(state.d_vec_d);
  if (state.d_vec_s) cudaFree(state.d_vec_s);
  if (state.d_md_d) cudaFree(state.d_md_d);
  if (state.d_md_s) cudaFree(state.d_md_s);
  state.d_cool = nullptr;
  state.d_cools = nullptr;
  state.d_vec_d = nullptr;
  state.d_vec_s = nullptr;
  state.d_md_d = nullptr;
  state.d_md_s = nullptr;
  if (state.pool_reserved > 0) {
    GlobalMemoryPool::dealloc(state.pool_reserved);
    state.pool_reserved = 0;
  }
  state.active = false;
  state.ready.store(-1);
}

bool ensure_handle() {
  if (state.handle) return true;
  if (cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking) !=
      cudaSuccess)
    return false;
  if (cublasCreate(&state.handle) != CUBLAS_STATUS_SUCCESS) {
    cudaStreamDestroy(state.stream);
    state.stream = nullptr;
    return false;
  }
  cublasSetPointerMode(state.handle, CUBLAS_POINTER_MODE_HOST);
  cublasSetStream(state.handle, state.stream);
  return true;
}

bool chunked_upload(char* dst, const char* src, size_t bytes, size_t chunk) {
  for (size_t off = 0; off < bytes; off += chunk) {
    if (state.abort_upload.load()) return false;
    const size_t n = (bytes - off < chunk) ? bytes - off : chunk;
    if (cudaMemcpyAsync(dst + off, src + off, n, cudaMemcpyHostToDevice,
                        state.stream) != cudaSuccess)
      return false;
    // Pageable copies are staged-synchronous anyway; the explicit sync just
    // bounds how much work the abort check can lag behind.
    if (cudaStreamSynchronize(state.stream) != cudaSuccess) return false;
  }
  return true;
}

// Runs on the worker thread: stream the copy, flip ready. Device buffers
// were already allocated synchronously in prefetch_ so that the XC cache
// auto-sizer (regenerate_partition's fgm detection, which reads live free
// VRAM at the first solve) sees our reservation and budgets around it.
void upload_worker(double* cool, float* cools) {
  const int Md = state.Md, nd = state.kknumd, ns = state.kknums;
  const size_t cool_bytes = (size_t)Md * nd * sizeof(double);
  const size_t cools_bytes = (size_t)Md * ns * sizeof(float);

  // The arrays are deliberately NOT host-pinned: cudaHostRegister on
  // 100+ MB holds the driver lock long enough to visibly stall XC kernel
  // launches running concurrently on other threads. The pageable staged
  // copy is slower for this one-time upload but contends in much shorter
  // slices; iterations use the CPU GEMV path until the copy lands.
  //
  // Copied in chunks so (a) a multi-GB upload checks abort_upload and lets
  // SCF teardown proceed instead of blocking on the full transfer, and
  // (b) each staged sync slice is short.
  const size_t CHUNK = 256 * 1024 * 1024;
  bool good = true;
  if (nd > 0)
    good = chunked_upload((char*)state.d_cool, (const char*)cool, cool_bytes,
                          CHUNK);
  if (good && ns > 0)
    good = chunked_upload((char*)state.d_cools, (const char*)cools,
                          cools_bytes, CHUNK);
  if (good) good = cudaStreamSynchronize(state.stream) == cudaSuccess;

  state.ready.store(good ? 1 : -1);
}

}  // namespace

/* Must be called before cool/cools are deallocated on the Fortran side:
 * the host arrays may be pinned with the driver. */
extern "C" void int3lu_gpu_invalidate_(void) { release_resources(); }

extern "C" void int3lu_gpu_finalize_(void) {
  release_resources();
  if (state.handle) {
    cublasDestroy(state.handle);
    state.handle = nullptr;
  }
  if (state.stream) {
    cudaStreamDestroy(state.stream);
    state.stream = nullptr;
  }
}

/* Kicks off the background upload of cool/cools. Called from the end of
 * int3mem, before the SCF loop touches the GPU. No-op if a prefetch is
 * already active for the current cool/cools. */
extern "C" void int3lu_gpu_prefetch_(double* cool, float* cools, int* md,
                                     int* kknumd, int* kknums) {
  if (state.active) return;

  const int Md = *md, nd = *kknumd, ns = *kknums;
  const size_t cool_bytes = (size_t)Md * nd * sizeof(double);
  const size_t cools_bytes = (size_t)Md * ns * sizeof(float);
  if (cool_bytes + cools_bytes < MIN_GPU_BYTES) return;

  size_t free_mem = 0, total_mem = 0;
  if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) return;
  const size_t need = cool_bytes + cools_bytes + nd * sizeof(double) +
                      ns * sizeof(float) + Md * (sizeof(double) + sizeof(float));

  // Mirror regenerate_partition's empirical headroom for the per-group XC
  // temporaries (partial densities, gradients, textures, ...) which are
  // allocated outside GlobalMemoryPool accounting: ~20% of free VRAM.
  // Undercutting it starves those allocations and crashes solve_opened
  // (observed on multiZn with a flat 256 MB margin).
  const size_t headroom =
      (free_mem / 5 > VRAM_MARGIN) ? free_mem / 5 : VRAM_MARGIN;
  if (need + headroom > free_mem) return;

  // Coordinate with the XC function-cache budget (GlobalMemoryPool was
  // already sized by the fgm auto-detect at SCF grid setup, before int3mem
  // runs). Memory the pool still plans to hand out is NOT ours to take:
  // claim unreserved VRAM first, and if that is not enough, carve the
  // shortfall out of the pool budget — the XC side then simply caches
  // fewer groups (partial caching) instead of both sides over-committing
  // the device.
  const size_t pool_remaining = GlobalMemoryPool::getFreeMemory();
  const size_t unclaimed = (free_mem > pool_remaining + headroom)
                               ? free_mem - pool_remaining - headroom
                               : 0;
  size_t shortfall = 0;
  if (need > unclaimed) {
    shortfall = need - unclaimed;
    if (GlobalMemoryPool::tryAlloc(shortfall) != 0) return;
    state.pool_reserved = shortfall;
  }

  if (!ensure_handle()) {
    if (state.pool_reserved > 0) {
      GlobalMemoryPool::dealloc(state.pool_reserved);
      state.pool_reserved = 0;
    }
    return;
  }

  state.Md = Md;
  state.kknumd = nd;
  state.kknums = ns;

  // Allocate the device buffers HERE, synchronously, not on the worker:
  // the XC function-cache auto-sizer reads live free VRAM at the first
  // solve and must see this reservation, or both sides over-commit the
  // card (observed as a segfault on multiZn: 5.5 GB cool + 5.5 GB XC cache
  // on a 12 GB device). The XC side handles a smaller budget gracefully
  // via its partial-caching path.
  bool good = true;
  if (nd > 0) {
    good = good && cudaMalloc(&state.d_cool, cool_bytes) == cudaSuccess;
    good = good && cudaMalloc(&state.d_vec_d, nd * sizeof(double)) == cudaSuccess;
  }
  if (ns > 0) {
    good = good && cudaMalloc(&state.d_cools, cools_bytes) == cudaSuccess;
    good = good && cudaMalloc(&state.d_vec_s, ns * sizeof(float)) == cudaSuccess;
  }
  good = good && cudaMalloc(&state.d_md_d, Md * sizeof(double)) == cudaSuccess;
  good = good && cudaMalloc(&state.d_md_s, Md * sizeof(float)) == cudaSuccess;
  if (!good) {
    release_resources();
    return;
  }

  state.active = true;
  state.ready.store(0);
  state.worker = std::thread(upload_worker, cool, cools);
}

/* ok=1 when the GPU path is active for int3lu_gpu_rc_/int3lu_gpu_terms_;
 * ok=0 means use the CPU path for this call. Also (re)starts the prefetch
 * in case int3mem's hook was bypassed. */
extern "C" void int3lu_gpu_ensure_(double* cool, float* cools, int* md,
                                   int* kknumd, int* kknums, int* ok) {
  if (!state.active) int3lu_gpu_prefetch_(cool, cools, md, kknumd, kknums);
  *ok = (state.active && state.ready.load() == 1) ? 1 : 0;
}

/* Rc_d = cool * rho_d ; Rc_s = cools * rho_s (both 'N'). */
extern "C" void int3lu_gpu_rc_(double* rho_d, float* rho_s, double* rc_d,
                               float* rc_s) {
  const double oned = 1.0, zerod = 0.0;
  const float onef = 1.0f, zerof = 0.0f;
  const int Md = state.Md;
  if (state.kknumd > 0) {
    cudaMemcpyAsync(state.d_vec_d, rho_d, state.kknumd * sizeof(double),
                    cudaMemcpyHostToDevice, state.stream);
    cublasDgemv(state.handle, CUBLAS_OP_N, Md, state.kknumd, &oned,
                state.d_cool, Md, state.d_vec_d, 1, &zerod, state.d_md_d, 1);
    cudaMemcpyAsync(rc_d, state.d_md_d, Md * sizeof(double),
                    cudaMemcpyDeviceToHost, state.stream);
  }
  if (state.kknums > 0) {
    cudaMemcpyAsync(state.d_vec_s, rho_s, state.kknums * sizeof(float),
                    cudaMemcpyHostToDevice, state.stream);
    cublasSgemv(state.handle, CUBLAS_OP_N, Md, state.kknums, &onef,
                state.d_cools, Md, state.d_vec_s, 1, &zerof, state.d_md_s, 1);
    cudaMemcpyAsync(rc_s, state.d_md_s, Md * sizeof(float),
                    cudaMemcpyDeviceToHost, state.stream);
  }
  cudaStreamSynchronize(state.stream);
}

/* terms_d = cool^T * af ; terms_s = cools^T * af_s (both 'T'). */
extern "C" void int3lu_gpu_terms_(double* af, float* af_s, double* terms_d,
                                  float* terms_s) {
  const double oned = 1.0, zerod = 0.0;
  const float onef = 1.0f, zerof = 0.0f;
  const int Md = state.Md;
  if (state.kknumd > 0) {
    cudaMemcpyAsync(state.d_md_d, af, Md * sizeof(double),
                    cudaMemcpyHostToDevice, state.stream);
    cublasDgemv(state.handle, CUBLAS_OP_T, Md, state.kknumd, &oned,
                state.d_cool, Md, state.d_md_d, 1, &zerod, state.d_vec_d, 1);
    cudaMemcpyAsync(terms_d, state.d_vec_d, state.kknumd * sizeof(double),
                    cudaMemcpyDeviceToHost, state.stream);
  }
  if (state.kknums > 0) {
    cudaMemcpyAsync(state.d_md_s, af_s, Md * sizeof(float),
                    cudaMemcpyHostToDevice, state.stream);
    cublasSgemv(state.handle, CUBLAS_OP_T, Md, state.kknums, &onef,
                state.d_cools, Md, state.d_md_s, 1, &zerof, state.d_vec_s, 1);
    cudaMemcpyAsync(terms_s, state.d_vec_s, state.kknums * sizeof(float),
                    cudaMemcpyDeviceToHost, state.stream);
  }
  cudaStreamSynchronize(state.stream);
}
