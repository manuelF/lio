# Optimization: Dynamic Task Scheduling (OpenMP Tasks)

## Summary
The current implementation of `Partition` (in `g2g/partition.cpp`) statically partitions PointGroups among CPU cores using `#pragma omp parallel for schedule(static)`.
*   Each core gets a fixed list of groups to process.
*   The CPU processes one group at a time serially (`inner_threads=1`).
*   Load imbalance occurs if group sizes vary significantly (e.g., one huge sphere vs many small cubes), leading to idle cores.
*   Rebalancing logic exists between iterations, but it's reactive and complex.

## Proposal
Switch to dynamic task scheduling using OpenMP Tasks.
1.  **Task Creation**: Create an OpenMP task for each PointGroup processing job.
2.  **Runtime Scheduling**: Let the OpenMP runtime schedule tasks to available worker threads dynamically.
3.  **Nested Parallelism**: For very large groups, use `omp taskloop` or nested parallelism (`inner_threads > 1`) to allow multiple cores to work on a single group if needed.

## Impact
*   **Load Balancing**: Automatically balances the workload across cores, reducing idle time.
*   **Scalability**: Handles heterogeneous group sizes gracefully.
*   **Throughput**: Improves overall parallel efficiency.

## Difficulty Assessment
**Low**

*   **Files to Modify**:
    *   `g2g/partition.cpp`: Specifically `Partition::solve` method.
    *   `g2g/common.h`: Check OpenMP includes.

*   **Correctness Impact**: **Low**. As long as `PointGroup`s are independent (which they are), order of execution doesn't matter. Accumulation into global results (if any) must be atomic or reduction-based.

*   **Complexity**: Standard OpenMP patterns.

## Sketch of Changes
1.  **Modify `Partition::solve` (`g2g/partition.cpp`)**:
    *   Replace the existing `#pragma omp parallel for` loop.
    *   Structure:
        ```cpp
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (auto& group : groups) {
                    #pragma omp task firstprivate(group)
                    {
                        group->solve(...);
                    }
                }
            }
            // Implicit barrier at end of single or parallel region waits for all tasks
        }
        ```
2.  **Nested Parallelism**:
    *   Check `omp_get_max_active_levels()`.
    *   If a group is flagged as `BIG`, inside `group->solve()` enable nested parallelism for BLAS calls (MKL uses OpenMP internally).

## Estimations
*   Speedup: 10-20% improvement in load balancing efficiency, especially for irregular grids.