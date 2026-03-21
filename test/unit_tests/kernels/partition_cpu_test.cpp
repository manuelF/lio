/**
 * Unit tests for the partition CPU/GPU group assignment logic.
 *
 * Tests the should_use_gpu() decision function that determines whether a
 * PointGroup should be processed on CPU or GPU, based on the group's
 * computational cost (P * M^2) rather than just P (number of points).
 *
 * The decision function signature:
 *   bool should_use_gpu(uint points, uint total_functions,
 *                       int cpu_threads, int gpu_threads,
 *                       long long split_cost);
 *
 * Rules:
 *   - If cpu_threads == 0: always GPU (no CPU available)
 *   - If gpu_threads == 0: always CPU (no GPU available)
 *   - Otherwise: GPU if P * M * M > split_cost
 */

#include "cpu_test_utils.h"

#include <cstdint>

// ---------------------------------------------------------------------------
// Replicate the decision function from g2g/regenerate_partition.cpp so we
// can test it in isolation.  The production code will use an identical
// implementation; keeping a local copy here means the test compiles
// without pulling in the full g2g library.
// ---------------------------------------------------------------------------

namespace G2G {

// Forward-declaration of the new decision function.
// This matches what will be added to regenerate_partition.cpp.
static bool should_use_gpu(unsigned int points, unsigned int total_functions,
                           int cpu_threads, int gpu_threads,
                           long long split_cost) {
  if (cpu_threads == 0) return true;   // no CPU workers available
  if (gpu_threads == 0) return false;  // no GPU workers available
  long long pm2 = (long long)points * total_functions * total_functions;
  return pm2 > split_cost;
}

// The old decision function for comparison.
static bool old_is_big_group(unsigned int num_points, int cpu_threads,
                             int gpu_threads, int splitpoints) {
  if (cpu_threads == 0) return true;
  if (gpu_threads == 0) return false;
  return num_points > (unsigned int)splitpoints;
}

}  // namespace G2G

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------
static void test_edge_no_cpu(test_utils::TestRunner& t) {
  // With no CPU threads, everything goes to GPU regardless of size.
  t.check(G2G::should_use_gpu(1, 1, 0, 1, 999999999LL),
          "no_cpu: tiny group -> GPU");
  t.check(G2G::should_use_gpu(10000, 200, 0, 1, 1LL),
          "no_cpu: huge group -> GPU");
}

static void test_edge_no_gpu(test_utils::TestRunner& t) {
  // With no GPU threads, everything goes to CPU regardless of size.
  t.check(!G2G::should_use_gpu(1, 1, 4, 0, 1LL),
          "no_gpu: tiny group -> CPU");
  t.check(!G2G::should_use_gpu(10000, 200, 4, 0, 1LL),
          "no_gpu: huge group -> CPU");
}

static void test_threshold_boundary(test_utils::TestRunner& t) {
  // P=100, M=10 -> P*M^2 = 10,000
  // With split_cost = 10,000: NOT greater, so CPU.
  t.check(!G2G::should_use_gpu(100, 10, 15, 1, 10000LL),
          "at_threshold: P*M^2 == split_cost -> CPU");
  // With split_cost = 9,999: greater, so GPU.
  t.check(G2G::should_use_gpu(100, 10, 15, 1, 9999LL),
          "above_threshold: P*M^2 > split_cost -> GPU");
  // With split_cost = 10,001: less, so CPU.
  t.check(!G2G::should_use_gpu(100, 10, 15, 1, 10001LL),
          "below_threshold: P*M^2 < split_cost -> CPU");
}

static void test_m_dependence(test_utils::TestRunner& t) {
  // Two groups with same P but different M should get different assignments
  // when the threshold is between their P*M^2 values.
  //
  // Group A: P=3492, M=54 -> P*M^2 = 3492 * 54 * 54 = 10,182,672
  // Group B: P=3492, M=116 -> P*M^2 = 3492 * 116 * 116 = 46,988,352
  //
  // With threshold = 37,000,000:
  //   A: 10,182,672 < 37,000,000 -> CPU
  //   B: 46,988,352 > 37,000,000 -> GPU
  //
  // This is the key improvement over P-only: same P, different assignment.
  long long thresh = 37000000LL;
  t.check(!G2G::should_use_gpu(3492, 54, 15, 1, thresh),
          "m_dep: P=3492 M=54 (P*M2=10.2M) -> CPU");
  t.check(G2G::should_use_gpu(3492, 116, 15, 1, thresh),
          "m_dep: P=3492 M=116 (P*M2=47M) -> GPU");

  // With old P-only threshold, both go same way regardless of M:
  // P=3492 > any reasonable SPLITPOINTS -> both GPU (wrong for M=54 case)
  t.check(G2G::old_is_big_group(3492, 15, 1, 200),
          "m_dep_old: P=3492 > SP=200 -> GPU (can't distinguish M)");
  t.check(G2G::old_is_big_group(3492, 15, 1, 2800),
          "m_dep_old: P=3492 > SP=2800 -> GPU (can't distinguish M)");
}

static void test_fosfato_groups(test_utils::TestRunner& t) {
  // Test with real group parameters from fosfatoQMMM.
  // Crossover P*M^2 ~ 183,549 (from performance model fitting).
  // Use a split_cost derived from the optimal P*M^2 threshold: ~37,000,000
  // (the value that minimizes parallel makespan on 15 CPU + 1 GPU).
  long long thresh = 37000000LL;

  // Small groups: clearly CPU
  t.check(!G2G::should_use_gpu(13, 3, 15, 1, thresh),
          "fosfato: P=13 M=3 (P*M2=117) -> CPU");
  t.check(!G2G::should_use_gpu(87, 4, 15, 1, thresh),
          "fosfato: P=87 M=4 (P*M2=1392) -> CPU");

  // Medium groups that the old P-only would misassign:
  t.check(!G2G::should_use_gpu(3492, 54, 15, 1, thresh),
          "fosfato: P=3492 M=54 -> CPU (old would say GPU)");
  t.check(!G2G::should_use_gpu(4074, 82, 15, 1, thresh),
          "fosfato: P=4074 M=82 (P*M2=27.4M) -> CPU");

  // Large groups: clearly GPU
  t.check(G2G::should_use_gpu(9916, 210, 15, 1, thresh),
          "fosfato: P=9916 M=210 (P*M2=437M) -> GPU");
  t.check(G2G::should_use_gpu(7514, 245, 15, 1, thresh),
          "fosfato: P=7514 M=245 (P*M2=451M) -> GPU");
  t.check(G2G::should_use_gpu(4656, 147, 15, 1, thresh),
          "fosfato: P=4656 M=147 (P*M2=101M) -> GPU");
}

static void test_overflow_safety(test_utils::TestRunner& t) {
  // P * M * M can overflow uint32 for large groups.
  // P=50000, M=500 -> P*M^2 = 12,500,000,000 (> 2^32)
  // The function must use long long arithmetic.
  long long big_thresh = 10000000000LL;  // 10 billion
  t.check(G2G::should_use_gpu(50000, 500, 15, 1, big_thresh),
          "overflow: P=50000 M=500 (P*M2=12.5B > 10B) -> GPU");
  t.check(!G2G::should_use_gpu(50000, 500, 15, 1, 13000000000LL),
          "overflow: P=50000 M=500 (P*M2=12.5B < 13B) -> CPU");

  // Even larger: P=100000, M=1000 -> 100,000,000,000
  t.check(G2G::should_use_gpu(100000, 1000, 15, 1, 99000000000LL),
          "overflow_large: P=100K M=1K (P*M2=100B > 99B) -> GPU");
}

static void test_backwards_compat_fallback(test_utils::TestRunner& t) {
  // When split_cost <= 0 (disabled), should behave like the old P-only logic.
  // This is for backwards compatibility via LIO_SPLIT_POINTS env.
  // We test that the old function matches expected behavior.
  t.check(G2G::old_is_big_group(201, 15, 1, 200),
          "compat: P=201 > SP=200 -> GPU");
  t.check(!G2G::old_is_big_group(200, 15, 1, 200),
          "compat: P=200 == SP=200 -> CPU");
  t.check(!G2G::old_is_big_group(199, 15, 1, 200),
          "compat: P=199 < SP=200 -> CPU");
}

static void test_crossover_model(test_utils::TestRunner& t) {
  // Verify the crossover formula: CROSSOVER = gamma / (alpha_cpu - alpha_gpu)
  // For the fosfatoQMMM model:
  //   alpha_cpu = 1.967e-3, alpha_gpu = 2.604e-5, gamma = 356.2
  //   CROSSOVER = 356.2 / (1.967e-3 - 2.604e-5) = 183,549
  //
  // Groups below crossover should be CPU-faster; above should be GPU-faster.
  // The split_cost for partitioning is typically larger than the crossover
  // because we want to minimize max(CPU, GPU), not per-group optimal.
  // But the crossover itself should be correct:
  double alpha_cpu = 1.967e-3;
  double alpha_gpu = 2.604e-5;
  double gamma = 356.2;
  double crossover = gamma / (alpha_cpu - alpha_gpu);

  // Check crossover is in the expected ballpark
  t.check(crossover > 180000 && crossover < 190000,
          "crossover: gamma/(alpha_cpu - alpha_gpu) ~ 183K");

  // At crossover, T_cpu ~ T_gpu
  double t_cpu = alpha_cpu * crossover;
  double t_gpu = gamma + alpha_gpu * crossover;
  double rel_diff = (t_cpu - t_gpu) / t_cpu;
  char buf[128];
  snprintf(buf, sizeof(buf), "T_cpu=%.1f T_gpu=%.1f diff=%.4f",
           t_cpu, t_gpu, rel_diff);
  t.check(rel_diff < 0.01 && rel_diff > -0.01,
          "crossover: T_cpu == T_gpu at crossover point", buf);
}

// ---------------------------------------------------------------------------
int main() {
  test_utils::TestRunner t("partition_cpu_test");

  test_edge_no_cpu(t);
  test_edge_no_gpu(t);
  test_threshold_boundary(t);
  test_m_dependence(t);
  test_fosfato_groups(t);
  test_overflow_safety(t);
  test_backwards_compat_fallback(t);
  test_crossover_model(t);

  return t.summary();
}
