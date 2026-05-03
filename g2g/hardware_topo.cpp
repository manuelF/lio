#include "hardware_topo.h"
#include <algorithm>
#include <cstdio>
#include <unistd.h>

int count_cpu_list(const char* s) {
  int count = 0, num = -1, range_from = -1;
  for (const char* p = s; ; ++p) {
    if (*p >= '0' && *p <= '9') {
      num = (num < 0 ? 0 : num) * 10 + (*p - '0');
    } else if (*p == '-') {
      range_from = (num >= 0) ? num : 0;
      num = -1;
    } else {
      if (num >= 0) {
        if (range_from >= 0) { count += num - range_from + 1; range_from = -1; }
        else count++;
        num = -1;
      }
      if (*p == '\0' || *p == '\n') break;
    }
  }
  return (count > 0) ? count : 1;
}

int detect_physical_cores() {
  int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
  if (n <= 0) n = 1;
  FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
  if (f) {
    char buf[256] = {};
    bool ok = (fgets(buf, sizeof(buf), f) != nullptr);
    fclose(f);
    if (ok) {
      int ht = count_cpu_list(buf);
      if (ht >= 1 && ht <= n) return std::max(1, n / ht);
    }
  }
  return n;
}

int recommended_omp_threads(int phys) {
  return std::max(2, (phys * 3) / 4);
}

int recommended_blas_threads(int phys) {
  return std::max(1, phys / 2);
}
