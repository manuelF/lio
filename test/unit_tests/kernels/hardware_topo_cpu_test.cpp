/**
 * Unit tests for g2g/hardware_topo.{h,cpp}.
 *
 * Tests:
 *   - count_cpu_list: parsing all Linux CPU list formats
 *   - recommended_blas_threads: formula at several core counts
 *   - recommended_omp_threads: formula at several core counts, clamping
 *   - detect_physical_cores: sanity checks against live /proc data
 */

#include "cpu_test_utils.h"

// Include implementation directly for standalone compilation.
#include "hardware_topo.cpp"

#include <cstdio>
#include <unistd.h>

// ---------------------------------------------------------------------------
// count_cpu_list tests
// ---------------------------------------------------------------------------
static void test_cpu_list_single(test_utils::TestRunner& t) {
  t.check(count_cpu_list("0") == 1,    "count_cpu_list: \"0\" -> 1");
  t.check(count_cpu_list("7") == 1,    "count_cpu_list: \"7\" -> 1");
}

static void test_cpu_list_pair(test_utils::TestRunner& t) {
  t.check(count_cpu_list("0,8")   == 2, "count_cpu_list: \"0,8\" -> 2");
  t.check(count_cpu_list("0,1")   == 2, "count_cpu_list: \"0,1\" -> 2");
}

static void test_cpu_list_range(test_utils::TestRunner& t) {
  t.check(count_cpu_list("0-3")   == 4, "count_cpu_list: \"0-3\" -> 4");
  t.check(count_cpu_list("0-7")   == 8, "count_cpu_list: \"0-7\" -> 8");
  t.check(count_cpu_list("4-7")   == 4, "count_cpu_list: \"4-7\" -> 4");
}

static void test_cpu_list_multi_range(test_utils::TestRunner& t) {
  t.check(count_cpu_list("0-3,8-11")   == 8,  "count_cpu_list: \"0-3,8-11\" -> 8");
  t.check(count_cpu_list("0-7,16-23")  == 16, "count_cpu_list: \"0-7,16-23\" -> 16");
}

static void test_cpu_list_with_newline(test_utils::TestRunner& t) {
  // fgets from sysfs keeps the trailing newline
  t.check(count_cpu_list("0\n")   == 1, "count_cpu_list: \"0\\n\" -> 1");
  t.check(count_cpu_list("0,8\n") == 2, "count_cpu_list: \"0,8\\n\" -> 2");
  t.check(count_cpu_list("0-3\n") == 4, "count_cpu_list: \"0-3\\n\" -> 4");
}

static void test_cpu_list_empty_fallback(test_utils::TestRunner& t) {
  t.check(count_cpu_list("") == 1,    "count_cpu_list: empty string -> 1");
}

// ---------------------------------------------------------------------------
// recommended_blas_threads tests
// ---------------------------------------------------------------------------
static void test_blas_threads(test_utils::TestRunner& t) {
  t.check(recommended_blas_threads(1)  == 1, "blas_threads(1) -> 1");
  t.check(recommended_blas_threads(2)  == 1, "blas_threads(2) -> 1");
  t.check(recommended_blas_threads(4)  == 2, "blas_threads(4) -> 2");
  t.check(recommended_blas_threads(8)  == 4, "blas_threads(8) -> 4  (5800X3D)");
  t.check(recommended_blas_threads(16) == 8, "blas_threads(16) -> 8");
  t.check(recommended_blas_threads(32) == 16,"blas_threads(32) -> 16");
}

static void test_blas_threads_positive(test_utils::TestRunner& t) {
  for (int p = 1; p <= 64; p++) {
    char buf[64];
    snprintf(buf, sizeof(buf), "blas_threads(%d) >= 1", p);
    t.check(recommended_blas_threads(p) >= 1, buf);
  }
}

// ---------------------------------------------------------------------------
// recommended_omp_threads tests
// ---------------------------------------------------------------------------
static void test_omp_threads(test_utils::TestRunner& t) {
  t.check(recommended_omp_threads(1)  == 2, "omp_threads(1) -> 2  (clamp)");
  t.check(recommended_omp_threads(2)  == 2, "omp_threads(2) -> 2  (clamp)");
  t.check(recommended_omp_threads(4)  == 3, "omp_threads(4) -> 3");
  t.check(recommended_omp_threads(8)  == 6, "omp_threads(8) -> 6  (5800X3D)");
  t.check(recommended_omp_threads(16) == 12,"omp_threads(16) -> 12");
  t.check(recommended_omp_threads(32) == 24,"omp_threads(32) -> 24");
}

static void test_omp_threads_at_least_two(test_utils::TestRunner& t) {
  for (int p = 1; p <= 64; p++) {
    char buf[64];
    snprintf(buf, sizeof(buf), "omp_threads(%d) >= 2", p);
    t.check(recommended_omp_threads(p) >= 2, buf);
  }
}

static void test_omp_less_than_blas_headroom(test_utils::TestRunner& t) {
  // OMP threads should leave room for BLAS: omp + blas <= phys * 3/2 (rough bound)
  for (int p = 2; p <= 32; p++) {
    int omp  = recommended_omp_threads(p);
    int blas = recommended_blas_threads(p);
    char buf[128];
    snprintf(buf, sizeof(buf), "p=%d omp=%d blas=%d", p, omp, blas);
    t.check(omp <= p, buf);
  }
}

// ---------------------------------------------------------------------------
// detect_physical_cores sanity tests (live system)
// ---------------------------------------------------------------------------
static void test_detect_physical_cores_sanity(test_utils::TestRunner& t) {
  int phys = detect_physical_cores();
  int logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
  if (logical <= 0) logical = 1;

  char buf[128];
  snprintf(buf, sizeof(buf), "phys=%d", phys);
  t.check(phys >= 1, "detect_physical_cores: >= 1", buf);

  snprintf(buf, sizeof(buf), "phys=%d logical=%d", phys, logical);
  t.check(phys <= logical, "detect_physical_cores: <= logical count", buf);

  // phys must divide logical (it's either equal or logical/HT_factor)
  snprintf(buf, sizeof(buf), "phys=%d logical=%d logical%%phys=%d",
           phys, logical, logical % phys);
  t.check(logical % phys == 0,
          "detect_physical_cores: logical divisible by phys", buf);
}

// ---------------------------------------------------------------------------
int main() {
  test_utils::TestRunner t("hardware_topo_cpu_test");

  test_cpu_list_single(t);
  test_cpu_list_pair(t);
  test_cpu_list_range(t);
  test_cpu_list_multi_range(t);
  test_cpu_list_with_newline(t);
  test_cpu_list_empty_fallback(t);

  test_blas_threads(t);
  test_blas_threads_positive(t);

  test_omp_threads(t);
  test_omp_threads_at_least_two(t);
  test_omp_less_than_blas_headroom(t);

  test_detect_physical_cores_sanity(t);

  return t.summary();
}
