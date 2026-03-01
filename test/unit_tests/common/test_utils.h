#pragma once
// Shared utilities for CUDA kernel unit tests.
// Include this header in any test .cu file.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// CUDA_CHECK: abort on any CUDA error with file/line info
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
      fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                          \
              __FILE__, __LINE__, cudaGetErrorString(_err));                \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

// ---------------------------------------------------------------------------
// TestRunner: lightweight pass/fail tracker with console output
// ---------------------------------------------------------------------------
namespace test_utils {

struct TestRunner {
  int passed = 0;
  int failed = 0;
  const char* suite_name;

  explicit TestRunner(const char* name) : suite_name(name) {
    printf("=== %s ===\n", name);
  }

  // Record one result and print it immediately.
  void check(bool ok, const char* test_name, const char* detail = nullptr) {
    if (ok) {
      printf("  [PASS] %s\n", test_name);
      ++passed;
    } else {
      if (detail)
        printf("  [FAIL] %s  (%s)\n", test_name, detail);
      else
        printf("  [FAIL] %s\n", test_name);
      ++failed;
    }
  }

  // Print final summary and return an appropriate process exit code.
  int summary() const {
    printf("\n[%s] %s: %d passed, %d failed\n",
           failed == 0 ? "OK  " : "FAIL",
           suite_name, passed, failed);
    return failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
  }
};

} // namespace test_utils
