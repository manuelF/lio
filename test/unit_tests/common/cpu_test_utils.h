#pragma once
// Shared utilities for CPU-only unit tests (no CUDA dependency).
// Include this header in any *_cpu_test.cpp file.

#include <cstdio>
#include <cstdlib>

namespace test_utils {

struct TestRunner {
  int passed = 0;
  int failed = 0;
  const char* suite_name;

  explicit TestRunner(const char* name) : suite_name(name) {
    printf("=== %s ===\n", name);
  }

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

  int summary() const {
    printf("\n[%s] %s: %d passed, %d failed\n",
           failed == 0 ? "OK  " : "FAIL",
           suite_name, passed, failed);
    return failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
  }
};

} // namespace test_utils
