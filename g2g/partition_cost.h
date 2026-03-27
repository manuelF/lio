#ifndef PARTITION_COST_H
#define PARTITION_COST_H

#include <vector>

namespace G2G {

// GPU hardware properties for performance model estimation.
// Populated once in g2g_init_() from cudaGetDeviceProperties.
struct GPUHardware {
  int sm_count;    // multiProcessorCount
  int clock_mhz;   // clockRate / 1000
  int major, minor; // compute capability
  int fp32_cores;   // total FP32 cores (sm_count * cores_per_sm)
  bool valid;       // false if no GPU or properties not yet queried
};

// FP32 CUDA cores per SM, by compute capability.
int cores_per_sm(int major, int minor);

// Estimate GPU/CPU speed ratio from hardware properties.
// Returns how many times faster the GPU kernel is vs one CPU core.
// Scaled from measured GTX 1080 baseline (75x at 2560 cores, 1733 MHz).
double estimate_speed_ratio(const GPUHardware& hw);

// Compute the optimal P*M^2 threshold for CPU/GPU work splitting.
// Uses LPT bin-packing simulation to minimize parallel makespan =
// max(CPU_bottleneck, GPU_total) given the actual group distribution.
//
// Parameters:
//   pm2_values  - P*M^2 cost for each group
//   n_cpu       - number of CPU threads
//   n_gpu       - number of GPU devices
//   speed_ratio - GPU/CPU speed ratio (from estimate_speed_ratio)
//   gpu_overhead - fixed overhead per GPU group in PM^2 units
//   out_makespan - if non-null, receives the best predicted makespan
//
// Returns the PM^2 threshold: groups with PM^2 > threshold go to GPU.
long long compute_optimal_split_cost(const std::vector<long long>& pm2_values,
                                     int n_cpu, int n_gpu,
                                     double speed_ratio,
                                     double gpu_overhead,
                                     double* out_makespan = nullptr);

// Convenience overload using default GPU overhead constant.
long long compute_optimal_split_cost(const std::vector<long long>& pm2_values,
                                     int n_cpu, int n_gpu,
                                     double speed_ratio,
                                     double* out_makespan = nullptr);

}  // namespace G2G

#endif  // PARTITION_COST_H
