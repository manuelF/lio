#include "partition_cost.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace G2G {

int cores_per_sm(int major, int minor) {
  switch (major) {
    case 2:
      return 32;   // Fermi
    case 3:
      return 192;  // Kepler
    case 5:
      return 128;  // Maxwell
    case 6:
      return (minor == 0) ? 64 : 128;   // Pascal (GP100 vs GP10x)
    case 7:
      return (minor == 0) ? 64 : 64;    // Volta / Turing
    case 8:
      return (minor == 0) ? 64 : 128;   // Ampere (A100 vs consumer)
    case 9:
      return 128;  // Hopper
    case 10:
      return 128;  // Blackwell
    default:
      return 128;  // future-proof guess
  }
}

double estimate_speed_ratio(const GPUHardware& hw) {
  if (!hw.valid) return 75.0;  // fallback if properties not available

  // Reference: GTX 1080 (SM 6.1) measured 75x per-core speedup.
  const double REF_CORES = 2560.0;   // 20 SMs * 128 cores
  const double REF_CLOCK = 1733.0;   // boost clock MHz
  const double REF_RATIO = 75.0;

  double gpu_throughput = (double)hw.fp32_cores * hw.clock_mhz;
  double ref_throughput = REF_CORES * REF_CLOCK;

  // Scale linearly with FP32 throughput.  This is approximate: the density
  // kernel is partially memory-bound, so actual scaling is sub-linear for
  // very fast GPUs.  But directionally correct and much better than a
  // fixed constant.
  double ratio = REF_RATIO * (gpu_throughput / ref_throughput);

  // Clamp to sane range: at minimum 10x (very old GPU), at most 2000x.
  if (ratio < 10.0) ratio = 10.0;
  if (ratio > 2000.0) ratio = 2000.0;
  return ratio;
}

// Default GPU overhead per group in PM^2 units (~19.2 us, dominated by
// CPU-side work so roughly constant across GPUs).
static const double DEFAULT_GPU_OVERHEAD = 180000.0;

long long compute_optimal_split_cost(const std::vector<long long>& pm2_values,
                                     int n_cpu, int n_gpu,
                                     double speed_ratio,
                                     double gpu_overhead,
                                     double* out_makespan) {
  // Trivial cases: only one device type available.
  if (n_cpu == 0 || n_gpu == 0 || pm2_values.empty()) return 0;

  std::vector<long long> sorted(pm2_values);
  std::sort(sorted.begin(), sorted.end());
  int N = (int)sorted.size();

  // Initial GPU total: all groups on GPU, nothing on CPU.
  // With multiple GPUs, groups are round-robin distributed, so effective
  // GPU time is total / n_gpu.
  double gpu_total = 0.0;
  for (int j = 0; j < N; j++)
    gpu_total += gpu_overhead + (double)sorted[j] / speed_ratio;
  gpu_total /= n_gpu;

  long long best_threshold = 0;
  double best_makespan = gpu_total;

  // Try each split: groups 0..split go to CPU, split+1..N-1 go to GPU.
  // Groups are sorted ascending by PM2.
  for (int split = 0; split < N; split++) {
    // Move group 'split' from GPU to CPU.
    gpu_total -= (gpu_overhead + (double)sorted[split] / speed_ratio) / n_gpu;

    // LPT bin-pack CPU groups (0..split) into n_cpu threads.
    // Assign largest-first to the least-loaded thread.
    std::vector<double> loads(n_cpu, 0.0);
    for (int i = split; i >= 0; i--) {
      int min_t = 0;
      for (int t = 1; t < n_cpu; t++)
        if (loads[t] < loads[min_t]) min_t = t;
      loads[min_t] += (double)sorted[i];
    }
    double cpu_bottleneck = 0.0;
    for (int t = 0; t < n_cpu; t++)
      if (loads[t] > cpu_bottleneck) cpu_bottleneck = loads[t];

    double makespan = std::max(cpu_bottleneck, gpu_total);
    if (makespan < best_makespan) {
      best_makespan = makespan;
      best_threshold = sorted[split];
    }
  }

  if (out_makespan) *out_makespan = best_makespan;
  return best_threshold;
}

long long compute_optimal_split_cost(const std::vector<long long>& pm2_values,
                                     int n_cpu, int n_gpu,
                                     double speed_ratio,
                                     double* out_makespan) {
  return compute_optimal_split_cost(pm2_values, n_cpu, n_gpu, speed_ratio,
                                    DEFAULT_GPU_OVERHEAD, out_makespan);
}

}  // namespace G2G
