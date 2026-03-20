# LIO CPU/GPU SPLITPOINTS Benchmark and Tuning Utility

## Table of Contents

1. [Purpose](#purpose)
2. [Background: How LIO Distributes Work](#background-how-lio-distributes-work)
3. [The Performance Model](#the-performance-model)
   - [Why P x M^2?](#why-p-x-m2)
   - [CPU Model](#cpu-model)
   - [GPU Model](#gpu-model)
   - [Crossover: When GPU Beats CPU](#crossover-when-gpu-beats-cpu)
4. [The SPLITPOINTS Problem](#the-splitpoints-problem)
   - [Why a Single Threshold is Imperfect](#why-a-single-threshold-is-imperfect)
   - [The Parallel Execution Constraint](#the-parallel-execution-constraint)
   - [The Cliff Effect](#the-cliff-effect)
5. [The Fitting Mathematics](#the-fitting-mathematics)
   - [CPU: Ordinary Least Squares Through the Origin](#cpu-ordinary-least-squares-through-the-origin)
   - [GPU: OLS with Intercept](#gpu-ols-with-intercept)
   - [Goodness of Fit (R^2)](#goodness-of-fit-r2)
   - [Data Filtering](#data-filtering)
6. [Parallel Execution Simulation](#parallel-execution-simulation)
   - [LPT Bin-Packing](#lpt-bin-packing)
   - [Objective Function](#objective-function)
   - [Search Strategy](#search-strategy)
7. [Reference Measurements (fosfatoQMMM)](#reference-measurements-fosfatommm)
8. [Using the Tool](#using-the-tool)
   - [Prerequisites](#prerequisites)
   - [Quick Start](#quick-start)
   - [Full Usage Reference](#full-usage-reference)
   - [Interpreting the Output](#interpreting-the-output)
   - [Using Saved Data](#using-saved-data)
   - [Sweep Mode](#sweep-mode)
9. [How the Measurement Works](#how-the-measurement-works)
10. [Limitations and Caveats](#limitations-and-caveats)
11. [Worked Example](#worked-example)

---

## Purpose

LIO splits DFT integration work between CPU threads and a GPU. The split is
controlled by a single integer threshold, `LIO_SPLIT_POINTS` (default: 200).
Groups of integration points with more than this many points go to the GPU;
smaller groups go to the CPU.

This threshold has a dramatic effect on performance. The default of 200 leaves
CPU threads 99% idle on typical systems, while a poorly chosen high value can
send a single huge group to a CPU thread and double the wall time.

`splitpoint_tune.py` measures actual per-group execution time on both devices,
fits a physical performance model, simulates the parallel execution, and
recommends the optimal SPLITPOINTS for your specific hardware and molecular
system.

---

## Background: How LIO Distributes Work

### The Partition System

At the start of each SCF cycle, LIO partitions all DFT numerical integration
points into **groups** (`PointGroup` objects). Each group contains:

- **P** = number of integration points (spatial grid points in a cube or sphere)
- **M** = number of overlapping basis functions (Gaussian-type orbitals whose
  extent reaches the group's spatial region)

Each group becomes either a `PointGroupCPU` or a `PointGroupGPU`, a permanent
decision made at partition creation time based solely on `P > SPLITPOINTS`.

### Thread Layout

LIO runs an OpenMP parallel region with `N = OMP_NUM_THREADS` total threads:

```
Thread 0             ... Thread (N-2)       Thread (N-1)
  CPU bin 0                CPU bin N-2         GPU bin 0
  [group, group, ...]      [group, ...]        [group, group, group, ...]
```

- Threads `0` through `N-2` are **CPU workers**. Each processes its assigned
  `PointGroupCPU` objects using scalar/SIMD CPU code.
- Thread `N-1` is the **GPU worker**. It processes all `PointGroupGPU` objects
  **sequentially**, launching CUDA kernels for each one.

After all threads finish (OMP barrier), the main thread merges per-thread
results into the global density matrix and forces.

### The Cost of Being Wrong

If SPLITPOINTS is too low (e.g., 200 on a 16-core machine):
- Nearly all groups go to the GPU.
- 15 CPU threads sit idle while the GPU thread works alone.
- Measured: CPU threads idle 99.3% of each SCF iteration.

If SPLITPOINTS is too high (e.g., 3200):
- A few large groups go to CPU, but one group may be much larger than the rest.
- That one CPU thread becomes the bottleneck (253 ms vs GPU's 132 ms).
- Wall time nearly doubles.

The sweet spot balances CPU and GPU finish times, utilizing all hardware.

---

## The Performance Model

### Why P x M^2?

The dominant computation in each group is the density kernel, which for each
of the P integration points evaluates:

```
rho(r) = sum_{i=1}^{M} sum_{j=1}^{i} P_ij * phi_i(r) * phi_j(r)
```

This is an M x M triangular sum at each of P points, giving O(P * M^2 / 2)
floating-point operations. The actual cost function used in LIO's partitioner
is:

```
cost = 10 * (P * M * (1 + M)) / 2 + MINCOST
```

which is proportional to P * M^2 for large M. Our performance model uses
`P * M^2` as the single independent variable, and the excellent R^2 values
(0.93-0.99) confirm this is the right scaling.

### CPU Model

The CPU executes the density kernel as a serial nested loop. There is
effectively no fixed overhead per group -- the cost is purely proportional
to the amount of computation:

```
T_cpu(P, M) = alpha_cpu * P * M^2
```

Where `alpha_cpu` is the time per unit of `P * M^2` in microseconds. This
is a single-parameter model fit through the origin.

**Why no intercept?** CPU groups have negligible setup cost. There is no kernel
launch, no memory transfer, no CUDA API call. The function call overhead is
sub-microsecond and is lost in timer noise for all but the tiniest groups.

**Measured value** (fosfatoQMMM, GTX 1080 + Intel i7-7700K):
```
alpha_cpu = 1.967e-03 us / (P * M^2)
R^2       = 0.985
```

### GPU Model

The GPU processes each group through a multi-step pipeline:

```
1. compute_functions()         -- evaluate basis functions (CUDA kernels)
2. get_rmm_input()             -- CPU: gather density matrix subblock
3. cudaMemcpy2DToArrayAsync()  -- upload to texture memory
4. gpu_compute_density()       -- CUDA kernel: density accumulation
5. gpu_update_rmm()            -- CUDA kernel: Fock matrix update
6. cudaStreamSynchronize()     -- wait for GPU
7. add_rmm_output()            -- CPU: scatter results back
```

Steps 1-3 and 7 represent **fixed overhead** that is roughly constant
regardless of group size. Steps 4-5 scale with the computation. This gives
a two-parameter model:

```
T_gpu(P, M) = gamma + alpha_gpu * P * M^2
```

Where:
- `gamma` = fixed overhead per group (microseconds). This includes CUDA kernel
  launch latency, cudaMemcpy setup, cudaMalloc/cudaFree from the memory pool,
  and CPU-side gather/scatter of the density matrix subblock.
- `alpha_gpu` = time per unit of `P * M^2` (microseconds). This is the
  marginal cost of additional computation, dominated by the GPU kernel.

**Measured values** (fosfatoQMMM, GTX 1080):
```
gamma     = 356 us   (fixed overhead per group)
alpha_gpu = 2.604e-05 us / (P * M^2)
R^2       = 0.926
```

### Crossover: When GPU Beats CPU

Setting `T_cpu = T_gpu` and solving for the crossover:

```
alpha_cpu * P * M^2 = gamma + alpha_gpu * P * M^2
(alpha_cpu - alpha_gpu) * P * M^2 = gamma
P * M^2 = gamma / (alpha_cpu - alpha_gpu)
```

This gives a single **crossover threshold in P*M^2 space**:

```
CROSSOVER = gamma / (alpha_cpu - alpha_gpu)
```

**Measured value:** `CROSSOVER = 183,549`

Below this value, the CPU is faster because the GPU's fixed 356 us overhead
dominates. Above this value, the GPU's 75.5x faster slope dominates.

#### Crossover in terms of P alone

Since SPLITPOINTS operates on P (not P*M^2), the effective crossover depends
on M:

```
P_crossover(M) = CROSSOVER / M^2
```

For groups with many basis functions (M = 100), the crossover P is only 18
points -- nearly all groups should go to GPU. For groups with few functions
(M = 3), the crossover P is 20,394 -- even very large groups are faster on
CPU.

This is a fundamental limitation of a P-only threshold: the optimal SPLITPOINTS
depends on M, which varies across groups.

---

## The SPLITPOINTS Problem

### Why a Single Threshold is Imperfect

The crossover formula `P_crossover = CROSSOVER / M^2` shows that the ideal
threshold is different for each group. A group with P=500 and M=3 (P*M^2 = 4500)
is firmly in CPU territory, while a group with P=500 and M=100 (P*M^2 = 5,000,000)
is firmly in GPU territory. Yet a single SPLITPOINTS must decide for both based
on P alone.

In practice, this is less severe than it sounds because M and P are correlated:
groups with many integration points tend to be spatially larger and overlap with
more basis functions. The benchmark tool accounts for this by testing every
actual P value as a candidate threshold against the measured per-group data.

### The Parallel Execution Constraint

The critical insight is that CPU and GPU run **in parallel**. The wall time
per SCF iteration is:

```
T_wall = max(T_cpu_bottleneck, T_gpu_total)
```

Where:
- `T_gpu_total` = sum of T_gpu for all GPU groups (they run sequentially on
  one GPU thread)
- `T_cpu_bottleneck` = the slowest CPU thread's total time (groups are
  distributed across N-1 CPU threads)

Minimizing the total per-group time (CPU + GPU) is **not** the right
objective. Instead, we want to minimize `max(CPU, GPU)` -- the parallel
makespan.

Moving a group from GPU to CPU:
- Decreases `T_gpu_total` by `T_gpu(group)`
- Increases `T_cpu_bottleneck` by approximately `T_cpu(group) / N_cpu_threads`
  (in the balanced case)

This is worthwhile as long as the GPU remains the bottleneck. It stops being
worthwhile when the CPU becomes the bottleneck -- that is the optimal split
point.

### The Cliff Effect

When a single large group is moved to CPU, it may not distribute evenly across
threads. If that group's CPU time exceeds the total time of all other CPU work
on any thread, it creates a **cliff**: one CPU thread becomes dramatically
slower than all others.

Example from fosfatoQMMM (the largest group: P=9261, M=365):
- `T_cpu = alpha_cpu * 9261 * 365^2 = 2,428,000 us = 2.4 seconds`
- This single group on one CPU thread would take 2.4 seconds, while the GPU
  finishes all remaining groups in ~130 ms.

The benchmark tool simulates this using LPT bin-packing (see below) to predict
cliffs before they happen.

---

## The Fitting Mathematics

### CPU: Ordinary Least Squares Through the Origin

The CPU model `T = alpha * x` where `x = P * M^2` has one parameter. The
optimal `alpha` minimizes the sum of squared residuals:

```
SSR = sum_i (y_i - alpha * x_i)^2
```

Taking the derivative and setting to zero:

```
d(SSR)/d(alpha) = -2 * sum_i x_i * (y_i - alpha * x_i) = 0
sum_i (x_i * y_i) = alpha * sum_i (x_i^2)
alpha = sum_i(x_i * y_i) / sum_i(x_i^2)
```

This is the standard no-intercept OLS estimator.

### GPU: OLS with Intercept

The GPU model `T = alpha * x + gamma` has two parameters. The normal equations:

```
| n       sum(x)   | | gamma |   | sum(y)   |
| sum(x)  sum(x^2) | | alpha | = | sum(x*y) |
```

Solving by Cramer's rule:

```
D     = n * sum(x^2) - sum(x)^2
alpha = (n * sum(x*y) - sum(x) * sum(y)) / D
gamma = (sum(y) - alpha * sum(x)) / n
```

### Goodness of Fit (R^2)

The coefficient of determination measures how much variance the model explains:

```
R^2 = 1 - SS_res / SS_tot

SS_res = sum_i (y_i - y_hat_i)^2     (residual sum of squares)
SS_tot = sum_i (y_i - y_bar)^2       (total sum of squares)
```

Where `y_hat_i` is the model prediction and `y_bar` is the mean of y.

- R^2 = 1.0: perfect fit
- R^2 = 0.0: model is no better than predicting the mean
- R^2 < 0.0: model is worse than the mean (bad model)

**Typical values:**
- CPU model: R^2 ~ 0.98 (excellent -- CPU timing is very predictable)
- GPU model: R^2 ~ 0.93 (good -- some variance from CUDA scheduling, memory
  pool state, and cache effects)

### Data Filtering

**CPU data:** Groups with T < 5 us are filtered out. At this timescale, the
`clock_gettime` timer resolution (~1 us) introduces significant relative error.
These tiny groups contribute negligible total time and would distort the fit.

**GPU data:** The first group (index 0) is skipped because it incurs cold-start
overhead: CUDA context initialization, first kernel JIT compilation, and initial
memory pool setup. This one-time cost would bias the intercept `gamma` upward.

---

## Parallel Execution Simulation

### LPT Bin-Packing

The tool simulates LIO's work distribution using the **Longest Processing Time
first (LPT)** heuristic, which matches what LIO's `compute_work_partition()`
does:

1. Sort all CPU groups by their CPU execution time, largest first.
2. Maintain a min-heap of (total_load, thread_id) for each CPU thread.
3. For each group (in decreasing size order), assign it to the least-loaded
   thread.

This is a well-known 4/3-approximation to the optimal makespan for the
multiprocessor scheduling problem. In practice, with many small groups and
15 threads, the result is near-optimal.

The simulation reports:
- **CPU/thr**: the time of the slowest (bottleneck) CPU thread
- **GPU**: the total sequential time of all GPU groups
- **Wall**: `max(CPU/thr, GPU)` -- the predicted iteration time
- **Bottleneck**: which device determines the wall time

### Objective Function

The tool minimizes:

```
minimize over SP:  max(T_cpu_bottleneck(SP), T_gpu_total(SP))
```

Where:
- `T_cpu_bottleneck(SP)` = max thread time after LPT bin-packing all groups
  with P <= SP
- `T_gpu_total(SP)` = sum of GPU times for all groups with P > SP

As SP increases:
- More groups move from GPU to CPU
- `T_gpu_total` decreases (less GPU work)
- `T_cpu_bottleneck` increases (more CPU work, possibly unevenly distributed)

The optimal SP is where these two curves cross, or more precisely, where
`max(CPU, GPU)` is minimized.

### Search Strategy

Rather than sweeping a fixed grid, the tool tests every actual P value that
appears in the system's groups as a candidate SPLITPOINTS, plus a set of
standard values (1, 50, 100, 200, 500, 1000, 2000, 3000, 5000). Since
SPLITPOINTS is a threshold on P, only the actual P values in the data create
distinct partitions. Testing all of them guarantees finding the global optimum
of the simulated objective.

---

## Reference Measurements (fosfatoQMMM)

These measurements were taken on 2026-03-20 on:
- **CPU:** Intel i7-7700K (4 cores / 8 threads, but tested with 16 OMP threads)
- **GPU:** NVIDIA GTX 1080 (Pascal, SM 6.1)
- **System:** fosfatoQMMM (34 QM atoms, closed-shell GGA, 25 SCF iterations)
- **Groups:** 165 total (131 cubes + 34 spheres)
- **Basis functions:** M ranges from 3 to 245 (median 61)
- **Points:** P ranges from 13 to 9261

### Fitted parameters

```
CPU:  T_cpu = 1.967e-03 * P * M^2  us          (R^2 = 0.985)
GPU:  T_gpu = 356.2 + 2.604e-05 * P * M^2  us  (R^2 = 0.926)

Crossover P*M^2 = 183,549
GPU slope speedup = 75.5x
```

### SPLITPOINTS sweep (actual wall time)

| SPLITPOINTS | Wall time | CPU max/iter | GPU/iter  | Bottleneck | Idle/iter |
|-------------|-----------|--------------|-----------|------------|-----------|
| 200         | 5.82 s    | 0.9 ms       | 134 ms    | GPU        | 133 ms    |
| 2800        | 5.49 s    | 91 ms        | 132 ms    | GPU        | 41 ms     |
| 3200        | 9.02 s    | 253 ms       | 132 ms    | CPU        | 122 ms    |

The sweet spot for this system is around SP=2800, where the CPU is loaded
enough to contribute meaningfully but no single group overwhelms a thread.
The cliff at SP=3200 occurs because one group (P=3590, M=83, T_cpu=105 ms)
exceeds the GPU's per-iteration time when it lands on an already-loaded
thread.

### Per-group crossover behavior

Near the crossover P*M^2 = 183,549:

| Group | P    | M   | P*M^2    | T_cpu (us) | T_gpu (us) | Faster |
|-------|------|-----|----------|------------|------------|--------|
| 85    | 120  | 24  | 69,120   | 108.5      | 139.4      | CPU    |
| 86    | 131  | 24  | 75,456   | 226.6      | 130.0      | GPU    |
| 87    | 177  | 25  | 110,625  | 316.6      | 139.1      | GPU    |
| 91    | 146  | 35  | 178,850  | 476.7      | 149.8      | GPU    |
| 92    | 156  | 36  | 202,176  | 303.9      | 145.3      | GPU    |

The transition from CPU-faster to GPU-faster happens around P*M^2 ~ 70,000
to 110,000 in practice, somewhat below the theoretical crossover of 183,549.
This is because the linear model slightly overestimates CPU time for
medium-sized groups (cache effects help the CPU more than the model predicts).

---

## Using the Tool

### Prerequisites

1. **Build LIO with both CPU and GPU support:**
   ```bash
   cd /path/to/lio
   make clean && make cuda=1 cpu=1 -j
   ```
   Both `cpu=1` and `cuda=1` are required. The tool runs liosolo twice: once
   with all groups on CPU, once with all groups on GPU.

2. **Source the environment:**
   ```bash
   source liohome.sh
   ```

3. **Have a test case ready.** You need the three standard LIO input files:
   - An input file (`.in`) with the `&lio` namelist
   - A coordinate file (`.xyz`)
   - A basis set file (`basis`)

4. **Python 3.6+** (no external packages required -- uses only the standard
   library).

### Quick Start

```bash
cd test/LIO_test/03_fosfatoQMMM
python3 ../../../g2g/bench/splitpoint_tune.py \
    -i fos.in -c fos.xyz -b basis
```

This will:
1. Run liosolo with all groups on CPU (~30-120 seconds depending on system)
2. Run liosolo with all groups on GPU (~5-15 seconds)
3. Print fitted model parameters, crossover analysis, and the recommended
   SPLITPOINTS value

To apply the recommendation:
```bash
export LIO_SPLIT_POINTS=<recommended_value>
```

Or add `LIO_SPLIT_POINTS=<value>` to your job script.

### Full Usage Reference

```
usage: splitpoint_tune.py [-h] --input INPUT --coords COORDS --basis BASIS
                          [--binary BINARY] [--sweep]
                          [--sweep-values SWEEP_VALUES] [--skip-measure]
                          [--cpu-data CPU_DATA] [--gpu-data GPU_DATA]
                          [--threads THREADS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `-i`, `--input` | Yes | LIO input file (e.g., `fos.in`) |
| `-c`, `--coords` | Yes | Coordinate file (e.g., `fos.xyz`) |
| `-b`, `--basis` | Yes | Basis set file (e.g., `basis`) |
| `--binary` | No | Path to liosolo binary. Auto-detected from `$LIOHOME/liosolo/liosolo`, PATH, or `../../../liosolo/liosolo`. |
| `--threads` | No | Total OMP_NUM_THREADS to simulate. Default: auto-detect from `$OMP_NUM_THREADS` or CPU count. The tool reserves 1 thread for GPU, so CPU threads = threads - 1. |
| `--sweep` | No | After model fitting, also run actual liosolo executions at several SPLITPOINTS values to validate the prediction with real wall times. |
| `--sweep-values` | No | Comma-separated list of SPLITPOINTS values to sweep (e.g., `200,1000,2000,3000`). Only used with `--sweep`. Default: auto-generated around the predicted optimum. |
| `--skip-measure` | No | Skip the liosolo runs; read pre-collected data from `--cpu-data` and `--gpu-data` instead. Useful for re-analyzing without re-running. |
| `--cpu-data` | No | File containing `[perfmodel]` output from an all-CPU run. Used with `--skip-measure`. |
| `--gpu-data` | No | File containing `[perfmodel]` output from an all-GPU run. Used with `--skip-measure`. |

### Interpreting the Output

The tool prints its analysis in six phases:

**Phase 1-2: Data Collection**
Runs liosolo twice (all-CPU, all-GPU). Reports the number of groups and wall
time for each run. If the all-CPU run is extremely slow (>200 seconds), your
system has many large groups that are expensive on CPU -- this is expected and
indicates the GPU provides large speedup.

**Phase 3: Model Fitting**
Reports `alpha_cpu`, `alpha_gpu`, `gamma`, and R^2 values. Key things to check:
- R^2 should be > 0.85 for both models. If significantly lower, the P*M^2
  model may not be appropriate for your system (unusual basis set or grid).
- `alpha_cpu / alpha_gpu` is the GPU's computational advantage. Typically
  50-150x depending on CPU and GPU hardware.
- `gamma` is the GPU's fixed overhead. Typically 200-500 us. Higher values
  indicate CUDA API overhead (cudaMalloc/cudaFree) or slow PCIe transfers.

**Phase 4: Crossover Analysis**
Reports which groups are faster on which device, and how many are "misassigned"
at the default SPLITPOINTS=200. The table shows groups near the crossover
point with their actual measured times on both devices.

**Phase 5: Optimal SPLITPOINTS**
The main result. Shows:
- The per-M crossover formula (`SP(M) = CROSSOVER / M^2`)
- Top 10 SPLITPOINTS candidates ranked by predicted wall time
- The optimal value and predicted improvement over the default

The table columns:
- **SP**: the SPLITPOINTS value tested
- **Wall us**: predicted per-iteration wall time = max(CPU bottleneck, GPU total)
- **CPU/thr us**: predicted time for the slowest CPU thread
- **GPU us**: predicted total GPU time (sum of all GPU groups)
- **Bottleneck**: which device determines the wall time

Look for the transition from GPU-bottlenecked to CPU-bottlenecked. The optimum
is at or just before this transition.

**Phase 6: Sweep (optional)**
If `--sweep` is passed, runs actual liosolo executions at several SPLITPOINTS
values. This validates the model's prediction with real wall times. The sweep
accounts for effects the model cannot capture: CUDA memory pool behavior, cache
warming, OMP scheduling overhead, and Fortran-side CPU work.

### Using Saved Data

To avoid re-running liosolo (which takes minutes), you can save the output
from a previous run and re-analyze it:

```bash
# Step 1: Collect data (run once)
source liohome.sh
cd test/LIO_test/03_fosfatoQMMM

# All-CPU run
LIO_SPLIT_POINTS=99999 \
  liosolo -i fos.in -c fos.xyz -b basis -v > /tmp/allcpu.txt 2>&1

# All-GPU run
LIO_SPLIT_POINTS=1 \
  liosolo -i fos.in -c fos.xyz -b basis -v > /tmp/allgpu.txt 2>&1

# Step 2: Analyze (instant, no liosolo needed)
python3 ../../../g2g/bench/splitpoint_tune.py \
    -i fos.in -c fos.xyz -b basis \
    --skip-measure \
    --cpu-data /tmp/allcpu.txt \
    --gpu-data /tmp/allgpu.txt
```

Note: The input files (`-i`, `-c`, `-b`) are still required even with
`--skip-measure` because the tool needs them for the sweep phase and for
locating the liosolo binary.

The `[perfmodel]` lines are emitted when `verbose = 4` is set in the LIO
input file. The tool automatically injects `verbose = 4` into a temporary
copy of the input file when running liosolo, but if you run liosolo manually,
you must add `verbose = 4` to the `&lio` namelist yourself:

```fortran
&lio
 natom = 34
 nsol  = 2954
 verbose = 4
 ...
&end
```

### Sweep Mode

The sweep validates the model's prediction by running liosolo at several
SPLITPOINTS values and measuring actual wall time:

```bash
# Auto-generated sweep points around the optimum
python3 ../../../g2g/bench/splitpoint_tune.py \
    -i fos.in -c fos.xyz -b basis --sweep

# Custom sweep points
python3 ../../../g2g/bench/splitpoint_tune.py \
    -i fos.in -c fos.xyz -b basis \
    --sweep --sweep-values 200,500,1000,2000,2500,3000,3500
```

Each sweep point runs a full liosolo execution (25 SCF iterations for
fosfatoQMMM), so sweeping 7 values takes ~7x the time of a single run.
The sweep also parses `[balance]` log lines to report per-iteration CPU
and GPU times.

---

## How the Measurement Works

### The [perfmodel] Instrumentation

The per-group timing data comes from instrumentation in `g2g/partition.cpp`
(the `Partition::solve()` function). When `verbose > 3`, on the **second**
SCF iteration, it prints one CSV line per group:

```
[perfmodel] idx,device,points,functions,cost,time_us
[perfmodel] 0,GPU,9261,365,617182570,4843.2
[perfmodel] 1,GPU,5932,245,178229370,1780.1
...
```

Fields:
- `idx`: group index (0..N-1)
- `device`: "CPU" or "GPU" (based on current SPLITPOINTS)
- `points`: P (number of integration points)
- `functions`: M (number of overlapping basis functions)
- `cost`: the cost function value `10 * P * M * (1+M) / 2 + MINCOST`
- `time_us`: measured wall-clock execution time in microseconds

**Why the second iteration?** The first SCF iteration has cold-start effects:
CUDA context initialization, first-time memory allocation, JIT compilation of
kernels, and initial basis function computation. The second iteration
represents steady-state performance.

### Two-Run Strategy

The tool runs liosolo twice with extreme SPLITPOINTS values:

1. **LIO_SPLIT_POINTS=99999** (all groups go to CPU): Measures `T_cpu` for
   every group. Since all groups are `PointGroupCPU`, there are no CUDA kernel
   launches. This isolates the pure CPU cost.

2. **LIO_SPLIT_POINTS=1** (all groups go to GPU): Measures `T_gpu` for every
   group. Since all groups are `PointGroupGPU`, each launches CUDA kernels.
   This captures the full GPU pipeline cost including fixed overhead.

Both runs produce the same groups with the same P and M values (the partition
geometry is deterministic), so the i-th row from each run corresponds to the
same physical group. This allows direct comparison.

---

## Limitations and Caveats

### Model assumes independent groups

The model treats each group's execution time as independent of other groups'
device assignment. In reality:
- When many groups are on GPU, the CUDA memory pool may fragment differently.
- CPU groups may compete for L3 cache and memory bandwidth.
- GPU memory caching (`free_global_memory`) changes the overhead pattern
  (cached groups skip `compute_functions()` and transposes after the first
  iteration).

The sweep mode captures these effects; the model prediction does not.

### Single-GPU only

The tool assumes one GPU (the common case). Multi-GPU setups with
`gpu_threads > 1` would need a more complex simulation that distributes
GPU groups across multiple GPU threads.

### P-only threshold

SPLITPOINTS operates on P alone, but the crossover depends on P*M^2. Groups
with the same P but different M may be better suited for different devices.
The tool finds the best single P threshold, but a P*M^2-based threshold
(or equivalently, a cost-based threshold) would be more precise. This
would require changes to LIO's partition code in `regenerate_partition.cpp`.

### Timer resolution

Groups with T < 5 us (CPU) are excluded from fitting because `clock_gettime`
resolution (~1 us) introduces large relative error. These groups are
negligible in total time contribution.

### System load sensitivity

The measurements assume a quiet system. Background processes competing for
CPU cores or GPU resources will distort the results. Run the benchmark on
an idle machine for best results.

### Hardware-specific

The fitted parameters (`alpha_cpu`, `alpha_gpu`, `gamma`) are specific to
your CPU, GPU, memory bandwidth, and CUDA driver version. Re-run the
benchmark after hardware changes, driver updates, or significant software
changes (new CUDA toolkit, different compiler flags).

### Molecule-specific

The group structure (P and M distributions) depends on the molecular system,
basis set, and grid parameters (`little_cube_size`, `max_function_exponent`,
`sphere_radius`). A SPLITPOINTS value tuned for one molecule may not be
optimal for a different molecule. For production QM/MM runs, tune on a
representative snapshot of your system.

---

## Worked Example

Here is the complete output from running the tool on fosfatoQMMM with
pre-collected data, annotated with interpretation:

```bash
$ python3 splitpoint_tune.py -i fos.in -c fos.xyz -b basis \
    --skip-measure --cpu-data /tmp/allcpu.txt --gpu-data /tmp/allgpu.txt \
    --threads 16
```

```
============================================================
Phase 3: Fitting performance models
============================================================

  CPU model: T_cpu = alpha_cpu x P x M^2
    alpha_cpu = 1.966688e-03 us per unit PxM^2     <-- CPU cost per unit work
    R^2    = 0.9845 (fitted on 128 groups)          <-- excellent fit

  GPU model: T_gpu = gamma + alpha_gpu x P x M^2
    gamma (fixed overhead) = 356.2 us               <-- ~0.36 ms per group just
    alpha_gpu = 2.603844e-05 us per unit PxM^2           to launch and transfer
    R^2    = 0.9262 (fitted on 164 groups)

  GPU slope speedup: alpha_cpu / alpha_gpu = 75.5x  <-- GPU kernel is 75x faster
```

Interpretation: The CPU model fits very well (R^2 = 0.98). The GPU model has
more variance (R^2 = 0.93) due to CUDA scheduling jitter, but is still good.
The GPU is 75x faster per unit of computation but pays 356 us fixed overhead.

```
============================================================
Phase 4: Crossover analysis
============================================================

  Crossover PxM^2 = 183549        <-- break-even point

  Summary: 86 groups faster on CPU, 79 faster on GPU
  With default SPLITPOINTS=200: 10 groups misassigned
```

Interpretation: 10 groups out of 165 are assigned to the wrong device with the
default threshold. These are groups near the crossover that happen to have P
around 120-200.

```
============================================================
Phase 5: Optimal SPLITPOINTS recommendation
============================================================

  Simulating parallel execution: 15 CPU threads + 1 GPU thread

        SP    Wall us   CPU/thr us     GPU us   Bottleneck
      2176     121008        41814     121008          GPU <-- best
      2026     122217        41814     122217          GPU
      2432     125044       125044     118762          CPU
                                      ^^^^^^^
                          At SP=2432, CPU becomes the bottleneck.
                          One large group overwhelms a CPU thread.

  Optimal SPLITPOINTS = 2176
  Predicted iter time: 121008 us (vs 132910 us at SP=200)
  Predicted improvement over default: +9.0%

  export LIO_SPLIT_POINTS=2176
```

Interpretation: The tool predicts SP=2176 minimizes the parallel makespan.
At this value, the GPU is still the bottleneck (121 ms) but the CPU is
contributing meaningfully (42 ms per thread) rather than sitting idle. Going
to SP=2432 flips the bottleneck to CPU (125 ms) because a large group can't
be evenly distributed.

The predicted +9% improvement is conservative because it doesn't account for
reduced GPU memory pressure and cache effects. The actual sweep (SP=2800)
measured ~6% improvement, consistent with the model's direction.
