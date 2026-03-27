/**
 * Unit tests for the partition cost model (partition_cost.h/cpp).
 *
 * Tests:
 *   - cores_per_sm: known GPU architectures
 *   - estimate_speed_ratio: scaling, clamping, fallback
 *   - compute_optimal_split_cost: trivial cases, balanced split,
 *     threshold boundary, multi-GPU, fosfato-like workloads
 */

#include "cpu_test_utils.h"

// Include the implementation directly for standalone compilation.
#include "partition_cost.cpp"

#include <cmath>
#include <cstdio>
#include <vector>

using G2G::GPUHardware;

// ---------------------------------------------------------------------------
// cores_per_sm tests
// ---------------------------------------------------------------------------
static void test_cores_per_sm(test_utils::TestRunner& t) {
  t.check(G2G::cores_per_sm(2, 0) == 32, "cores_per_sm: Fermi 2.0 = 32");
  t.check(G2G::cores_per_sm(3, 5) == 192, "cores_per_sm: Kepler 3.5 = 192");
  t.check(G2G::cores_per_sm(5, 0) == 128, "cores_per_sm: Maxwell 5.0 = 128");
  t.check(G2G::cores_per_sm(6, 0) == 64, "cores_per_sm: Pascal GP100 6.0 = 64");
  t.check(G2G::cores_per_sm(6, 1) == 128,
          "cores_per_sm: Pascal GP10x 6.1 = 128");
  t.check(G2G::cores_per_sm(7, 0) == 64, "cores_per_sm: Volta 7.0 = 64");
  t.check(G2G::cores_per_sm(7, 5) == 64, "cores_per_sm: Turing 7.5 = 64");
  t.check(G2G::cores_per_sm(8, 0) == 64, "cores_per_sm: Ampere A100 8.0 = 64");
  t.check(G2G::cores_per_sm(8, 6) == 128,
          "cores_per_sm: Ampere consumer 8.6 = 128");
  t.check(G2G::cores_per_sm(9, 0) == 128, "cores_per_sm: Hopper 9.0 = 128");
  t.check(G2G::cores_per_sm(10, 0) == 128,
          "cores_per_sm: Blackwell 10.0 = 128");
  t.check(G2G::cores_per_sm(99, 0) == 128,
          "cores_per_sm: unknown arch = 128 fallback");
}

// ---------------------------------------------------------------------------
// estimate_speed_ratio tests
// ---------------------------------------------------------------------------
static void test_speed_ratio_fallback(test_utils::TestRunner& t) {
  GPUHardware invalid = {0, 0, 0, 0, 0, false};
  double ratio = G2G::estimate_speed_ratio(invalid);
  t.check(std::abs(ratio - 75.0) < 0.01,
          "speed_ratio: invalid hw returns 75.0 fallback");
}

static void test_speed_ratio_gtx1080(test_utils::TestRunner& t) {
  // GTX 1080: 2560 cores, 1733 MHz — reference card, should give ~75x
  GPUHardware gtx1080 = {20, 1733, 6, 1, 2560, true};
  double ratio = G2G::estimate_speed_ratio(gtx1080);
  char buf[128];
  snprintf(buf, sizeof(buf), "ratio=%.1f", ratio);
  t.check(std::abs(ratio - 75.0) < 1.0,
          "speed_ratio: GTX 1080 ~ 75x", buf);
}

static void test_speed_ratio_scaling(test_utils::TestRunner& t) {
  // A card with 2x the throughput of GTX 1080 should give ~2x the ratio.
  GPUHardware fast = {40, 1733, 8, 0, 5120, true};
  double ratio = G2G::estimate_speed_ratio(fast);
  char buf[128];
  snprintf(buf, sizeof(buf), "ratio=%.1f", ratio);
  t.check(ratio > 140.0 && ratio < 160.0,
          "speed_ratio: 2x throughput ~ 150x", buf);
}

static void test_speed_ratio_clamping(test_utils::TestRunner& t) {
  // Very weak GPU: should clamp to 10x minimum.
  GPUHardware weak = {1, 100, 2, 0, 32, true};
  double ratio = G2G::estimate_speed_ratio(weak);
  t.check(std::abs(ratio - 10.0) < 0.01,
          "speed_ratio: very weak GPU clamped to 10x");

  // Impossibly fast GPU: should clamp to 2000x.
  // Need enough throughput to exceed 2000x: 2000/75 * 2560*1733 ~ 118M
  GPUHardware monster = {500, 5000, 10, 0, 500 * 128, true};
  double ratio2 = G2G::estimate_speed_ratio(monster);
  t.check(std::abs(ratio2 - 2000.0) < 0.01,
          "speed_ratio: monster GPU clamped to 2000x");
}

// ---------------------------------------------------------------------------
// compute_optimal_split_cost tests
// ---------------------------------------------------------------------------
static void test_split_trivial_no_cpu(test_utils::TestRunner& t) {
  std::vector<long long> pm2 = {1000, 2000, 3000};
  long long thresh =
      G2G::compute_optimal_split_cost(pm2, 0, 1, 75.0);
  t.check(thresh == 0, "split: n_cpu=0 returns 0 (all GPU)");
}

static void test_split_trivial_no_gpu(test_utils::TestRunner& t) {
  std::vector<long long> pm2 = {1000, 2000, 3000};
  long long thresh =
      G2G::compute_optimal_split_cost(pm2, 4, 0, 75.0);
  t.check(thresh == 0, "split: n_gpu=0 returns 0 (all CPU)");
}

static void test_split_trivial_empty(test_utils::TestRunner& t) {
  std::vector<long long> pm2;
  long long thresh =
      G2G::compute_optimal_split_cost(pm2, 4, 1, 75.0);
  t.check(thresh == 0, "split: empty groups returns 0");
}

static void test_split_single_group(test_utils::TestRunner& t) {
  // One group: should go to whichever device is faster for it.
  // PM2 = 10M, speed_ratio=75, overhead=180K
  // GPU time: 180K + 10M/75 = 313K
  // CPU time: 10M
  // GPU is faster, so threshold should be 0 (keep it on GPU).
  std::vector<long long> pm2 = {10000000LL};
  double makespan = 0.0;
  long long thresh =
      G2G::compute_optimal_split_cost(pm2, 1, 1, 75.0, 180000.0, &makespan);
  char buf[128];
  snprintf(buf, sizeof(buf), "thresh=%lld makespan=%.0f", thresh, makespan);
  t.check(thresh == 0,
          "split: single large group stays on GPU", buf);
}

static void test_split_all_tiny(test_utils::TestRunner& t) {
  // Many tiny groups: overhead dominates, all should go to CPU.
  // PM2=100 each, overhead=180K, speed_ratio=75
  // GPU cost per group: 180K + 100/75 ~ 180K — much more than CPU cost of 100.
  std::vector<long long> pm2(50, 100LL);
  double makespan = 0.0;
  long long thresh = G2G::compute_optimal_split_cost(pm2, 4, 1, 75.0,
                                                      180000.0, &makespan);
  // All 50 groups have PM2=100, so threshold should be >= 100 (all on CPU).
  t.check(thresh >= 100,
          "split: 50 tiny groups all go to CPU");
}

static void test_split_balanced_workload(test_utils::TestRunner& t) {
  // Mix of small and large groups.
  // 10 small (PM2=1K) + 5 large (PM2=50M)
  // With speed_ratio=75, overhead=180K:
  //   Large GPU cost: 180K + 50M/75 = 847K per group
  //   Large CPU cost: 50M per group
  //   Small GPU cost: 180K + 1K/75 ~ 180K per group (overhead-dominated)
  //   Small CPU cost: 1K per group
  // Small groups should go to CPU (overhead too high for GPU).
  // Large groups should go to GPU (75x speedup).
  std::vector<long long> pm2;
  for (int i = 0; i < 10; i++) pm2.push_back(1000LL);
  for (int i = 0; i < 5; i++) pm2.push_back(50000000LL);

  double makespan = 0.0;
  long long thresh = G2G::compute_optimal_split_cost(pm2, 4, 1, 75.0,
                                                      180000.0, &makespan);
  char buf[128];
  snprintf(buf, sizeof(buf), "thresh=%lld", thresh);
  // Threshold should be between small and large: small on CPU, large on GPU.
  // thresh >= 1000 means PM2=1000 groups stay on CPU (PM2 > thresh → GPU).
  t.check(thresh >= 1000LL && thresh < 50000000LL,
          "split: small->CPU large->GPU", buf);
}

static void test_split_makespan_decreases(test_utils::TestRunner& t) {
  // Verify that splitting reduces makespan vs all-GPU or all-CPU.
  // 20 groups with varying costs.
  std::vector<long long> pm2;
  for (int i = 1; i <= 20; i++) pm2.push_back((long long)i * 1000000LL);

  double makespan_split = 0.0;
  G2G::compute_optimal_split_cost(pm2, 8, 1, 75.0, 180000.0, &makespan_split);

  // All-GPU makespan: sum(overhead + pm2/speed_ratio)
  double all_gpu = 0.0;
  for (size_t i = 0; i < pm2.size(); i++)
    all_gpu += 180000.0 + (double)pm2[i] / 75.0;

  // All-CPU makespan: max bin with LPT on 8 threads
  // Rough estimate: total / 8
  double total_cpu = 0.0;
  for (size_t i = 0; i < pm2.size(); i++) total_cpu += (double)pm2[i];
  double all_cpu_approx = total_cpu / 8.0;

  char buf[256];
  snprintf(buf, sizeof(buf), "split=%.0f gpu=%.0f cpu_approx=%.0f",
           makespan_split, all_gpu, all_cpu_approx);
  t.check(makespan_split < all_gpu && makespan_split < all_cpu_approx,
          "split: makespan < all-GPU and < all-CPU", buf);
}

static void test_split_multi_gpu(test_utils::TestRunner& t) {
  // With 2 GPUs, more groups can go to GPU profitably.
  std::vector<long long> pm2;
  for (int i = 0; i < 10; i++) pm2.push_back(10000000LL);

  double makespan_1gpu = 0.0, makespan_2gpu = 0.0;
  G2G::compute_optimal_split_cost(pm2, 4, 1, 75.0, 180000.0, &makespan_1gpu);
  G2G::compute_optimal_split_cost(pm2, 4, 2, 75.0, 180000.0, &makespan_2gpu);

  char buf[128];
  snprintf(buf, sizeof(buf), "1gpu=%.0f 2gpu=%.0f", makespan_1gpu,
           makespan_2gpu);
  t.check(makespan_2gpu <= makespan_1gpu,
          "split: 2 GPUs <= 1 GPU makespan", buf);
}

static void test_split_threshold_is_group_value(test_utils::TestRunner& t) {
  // The returned threshold must be one of the actual PM2 values (it's the
  // largest group assigned to CPU).
  std::vector<long long> pm2 = {100LL, 500LL, 5000LL, 50000LL, 500000LL};
  long long thresh = G2G::compute_optimal_split_cost(pm2, 4, 1, 75.0);
  bool is_member = false;
  for (size_t i = 0; i < pm2.size(); i++) {
    if (pm2[i] == thresh) {
      is_member = true;
      break;
    }
  }
  // thresh could also be 0 (all on GPU)
  t.check(thresh == 0 || is_member,
          "split: threshold is 0 or a group PM2 value");
}

static void test_split_fosfato_like(test_utils::TestRunner& t) {
  // Realistic workload inspired by fosfatoQMMM:
  // ~50 CPU groups (PM2 100-1M), ~45 GPU groups (PM2 10M-500M)
  // 15 CPU threads, 1 GPU, speed_ratio ~75
  std::vector<long long> pm2;
  // Small CPU-bound groups
  for (int i = 0; i < 30; i++) pm2.push_back(100LL + i * 30000LL);
  // Medium groups (boundary region)
  for (int i = 0; i < 20; i++) pm2.push_back(1000000LL + i * 2000000LL);
  // Large GPU-bound groups
  for (int i = 0; i < 45; i++) pm2.push_back(50000000LL + i * 10000000LL);

  double makespan = 0.0;
  long long thresh = G2G::compute_optimal_split_cost(pm2, 15, 1, 75.0,
                                                      180000.0, &makespan);
  char buf[128];
  snprintf(buf, sizeof(buf), "thresh=%lld makespan=%.0f", thresh, makespan);
  // The threshold should separate small/medium from large groups.
  // Not all on GPU (thresh > 0) and not all on CPU.
  t.check(thresh > 0 && thresh < 500000000LL,
          "split: fosfato-like gives sensible threshold", buf);
}

static void test_split_speed_ratio_sensitivity(test_utils::TestRunner& t) {
  // Higher speed_ratio should move the threshold down (more groups to GPU).
  std::vector<long long> pm2;
  for (int i = 1; i <= 20; i++) pm2.push_back((long long)i * 1000000LL);

  long long thresh_slow = G2G::compute_optimal_split_cost(pm2, 4, 1, 20.0);
  long long thresh_fast = G2G::compute_optimal_split_cost(pm2, 4, 1, 200.0);

  char buf[128];
  snprintf(buf, sizeof(buf), "slow=%lld fast=%lld", thresh_slow, thresh_fast);
  // Faster GPU → more groups profitable on GPU → lower threshold.
  t.check(thresh_fast <= thresh_slow,
          "split: faster GPU -> lower threshold", buf);
}

static void test_split_overhead_sensitivity(test_utils::TestRunner& t) {
  // Higher overhead should move the threshold up (fewer groups to GPU).
  std::vector<long long> pm2;
  for (int i = 1; i <= 20; i++) pm2.push_back((long long)i * 1000000LL);

  long long thresh_low =
      G2G::compute_optimal_split_cost(pm2, 4, 1, 75.0, 10000.0);
  long long thresh_high =
      G2G::compute_optimal_split_cost(pm2, 4, 1, 75.0, 500000.0);

  char buf[128];
  snprintf(buf, sizeof(buf), "low_oh=%lld high_oh=%lld", thresh_low,
           thresh_high);
  // Higher overhead → fewer profitable GPU groups → higher threshold.
  t.check(thresh_high >= thresh_low,
          "split: higher overhead -> higher threshold", buf);
}

// ---------------------------------------------------------------------------
int main() {
  test_utils::TestRunner t("partition_cost_cpu_test");

  test_cores_per_sm(t);
  test_speed_ratio_fallback(t);
  test_speed_ratio_gtx1080(t);
  test_speed_ratio_scaling(t);
  test_speed_ratio_clamping(t);
  test_split_trivial_no_cpu(t);
  test_split_trivial_no_gpu(t);
  test_split_trivial_empty(t);
  test_split_single_group(t);
  test_split_all_tiny(t);
  test_split_balanced_workload(t);
  test_split_makespan_decreases(t);
  test_split_multi_gpu(t);
  test_split_threshold_is_group_value(t);
  test_split_fosfato_like(t);
  test_split_speed_ratio_sensitivity(t);
  test_split_overhead_sensitivity(t);

  return t.summary();
}
