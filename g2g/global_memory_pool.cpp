#include "global_memory_pool.h"
#include <cassert>
#include <cstdio>

// Only include it to cudaGetMemoryInfo to accurately measure the device memory.
#include "cuda_includes.h"

// TryAlloc return 1 if error, 0 if success
int GlobalMemoryPool::tryAlloc(size_t size) {
  if (!_init) init();
  int current_device = 0;
#if GPU_KERNELS
  cudaGetDevice(&current_device);
#endif
  if (_freeGlobalMemory[current_device] < size) return 1;
  _freeGlobalMemory[current_device] -= size;
  return 0;
}

void GlobalMemoryPool::dealloc(size_t size, int device) {
  if (!_init) init();
  _freeGlobalMemory[device] += size;
}

void GlobalMemoryPool::dealloc(size_t size) {
  if (!_init) init();
  int current_device = 0;
#if GPU_KERNELS
  cudaGetDevice(&current_device);
#endif
  assert(_freeGlobalMemory[current_device] > size);
  _freeGlobalMemory[current_device] += size;
}

void GlobalMemoryPool::init(double free_global_memory) {
  // Re-apply the budget on every call, not just the first: each partition
  // rebuild computes a fresh fgm for its group sizes, and pinning to the first
  // (SCF) partition starves later ones -- the TD merged cube would fail tryAlloc
  // and recompute its functions every step. Safe because init() runs after
  // Partition::clear() and before any new tryAlloc.
#if GPU_KERNELS
  int previous_device;
  cudaGetDevice(&previous_device);
  int gpu_count = cudaGetGPUCount();
  for (int i = 0; i < gpu_count; i++) {
    size_t free_memory = 0, total_memory = 0;
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    cudaGetMemoryInfo(free_memory, total_memory);
    cudaDeviceSynchronize();
    double free_factor = free_global_memory;

    if (free_factor > 1.0f) free_factor = 1.0f;
    if (free_factor < 0.0f) free_factor = 0.0f;
    _freeFactor = free_factor;

    size_t budget =
        static_cast<size_t>(static_cast<double>(free_memory) * _freeFactor);
    if (!_init) {
      _freeGlobalMemory.push_back(budget);
      _totalGlobalMemory.push_back(total_memory);
    } else {
      _freeGlobalMemory[i] = budget;
      _totalGlobalMemory[i] = total_memory;
    }
  }
  cudaSetDevice(previous_device);
#else
  if (!_init) {
    _totalGlobalMemory.push_back(0);
    _freeGlobalMemory.push_back(0);
  }
  _freeFactor = 0.0f;
#endif
  _init = true;
}

size_t GlobalMemoryPool::getFreeMemory() {
  if (!_init) init();
  size_t free_memory = 0;
#if GPU_KERNELS
  int current_device;
  cudaGetDevice(&current_device);
  free_memory = _freeGlobalMemory[current_device];
#endif
  return free_memory;
}

std::vector<size_t> GlobalMemoryPool::_freeGlobalMemory;
std::vector<size_t> GlobalMemoryPool::_totalGlobalMemory;
float GlobalMemoryPool::_freeFactor = 0.8;
bool GlobalMemoryPool::_init = false;
