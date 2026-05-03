#ifndef HARDWARE_TOPO_H
#define HARDWARE_TOPO_H

// Hardware topology helpers — no CUDA, no OpenMP, no G2G dependencies.
// Used by g2g_init_() for auto-tuning OMP/BLAS thread counts.

// Parse a Linux CPU list string (e.g. "0", "0,8", "0-3", "0-3,8-11") and
// return the number of CPUs it contains.  Returns 1 on empty or bad input.
int count_cpu_list(const char* s);

// Return the physical (non-HT) core count by reading the HT-siblings topology
// file.  Falls back to hardware_concurrency() if the sysfs file is unavailable.
int detect_physical_cores();

// Recommended OMP thread count when int3lu/g2g overlap is active.
// Formula: max(2, phys*3/4), calibrated on 5800X3D (8 phys → 6).
int recommended_omp_threads(int phys);

// Recommended BLAS thread count for the int3lu section during overlap.
// Formula: max(1, phys/2), calibrated on 5800X3D (8 phys → 4).
int recommended_blas_threads(int phys);

#endif  // HARDWARE_TOPO_H
