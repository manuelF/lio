---
name: td-multi-stream-gpu-workers
description: Multi-thread / multi-stream GPU driver (1 GPU, N OpenMP workers, per-thread default stream) tested on TD chloride open-shell. REJECTED — empirically 3-4× SLOWER than single-thread baseline due to host-side contention.
status: REJECTED — net-negative on RTX 3080 Ti + CUDA 13.1 + nvcc 12.0
metadata:
  type: project
  workload: 07_TDDFTHCL/chloride.in (open shell, propagator=2, ntdstep=5000)
  date: 2026-05-21
  hardware: RTX 3080 Ti (Ampere SM 8.6), Ryzen 7 5800X3D
---

# Multi-thread / multi-stream GPU driver — REJECTED

## Hypothesis

TDDFT propagation has GPU busy fraction only ~55% (see [[td-inter-step-gap]]).
Per-step GPU work is 8 groups × ~10 tiny kernels (median density kernel 13 µs).
**5 of 8 distinct group sizes use only 4–23 SMs out of 80** — under-occupied.

Hypothesis: assign multiple OpenMP worker threads to drive the same GPU, each
through its own CUDA stream + cuBLAS handle. Per-thread default streams
(NVCC `--default-stream per-thread`) make each host thread see a private,
non-blocking stream-0. Two threads launching kernels on two streams should
let small-grid groups run concurrently and the GPU sit busier.

## What was tried

Patch (kept in branch under env-var gate, default off):

- `g2g/Makefile.cuda`: add `NVCCFLAGS += --default-stream per-thread`
  (benign for single-thread runs — the main thread's stream-0 is just renamed).
- `g2g/init.cpp`: env `LIO_GPU_THREADS` (default 1, cap 4) multiplies
  `gpu_threads` by that count for a single device; build a `gpu_thread_device[]`
  map so `Partition::solve()` can `cudaSetDevice()` the right device per worker.
- `g2g/partition.cpp`: `cudaDeviceSynchronize()` before and after the OMP
  parallel-for so the pre-parallel uploads (`s_global_rdm_dev`) and
  post-parallel download of `s_global_fock_dev` are visible across threads
  (per-thread default streams don't auto-sync with the legacy default).
- `g2g/cuda/iteration.cu`: cuBLAS handle changed from `static` singleton to
  `thread_local`, with a small registry+mutex so shutdown can destroy all of
  them. cuBLAS automatically targets the calling thread's stream in
  per-thread-default mode, so no `cublasSetStream` calls are needed.

## Numbers — `chloride.in` 5000-step (open-shell), no overlap mode

| LIO_GPU_THREADS | Wall (s) | vs baseline |
|---|---|---|
| 1 (current default) | 5.95 | 1.00× |
| **2** | **18.27** | **3.07× slower** |
| 3 | 22.70 | 3.81× slower |

With `LIO_OVERLAP_INT3LU_G2G=1`:

| LIO_GPU_THREADS | Wall (s) |
|---|---|
| 1 | 4.65 (matches pre-patch baseline) |
| 2 | hang/timeout (> 60 s, OMP nested-team interaction with overlap path) |

SCF (fosfato, regression check): 1.92s at `LIO_GPU_THREADS=1` — same as
pre-patch baseline.

## Why it fails empirically

`nsys` trace at `LIO_GPU_THREADS=2` (5 s capture):
- **GPU busy fraction collapses to 12.2%** (baseline was 55%).
- Median per-kernel gap on each worker stream **balloons to 13–18 µs**
  (baseline was 0.7 µs back-to-back).
- Both worker streams (13 and 20) accumulate ~8000 kernels each; cuBLAS
  spawns an internal stream (7) for splitK/dgmm internals, doubling driver
  work.

The CPU-side overhead per kernel launch jumps from ~5 µs (single thread) to
~15–18 µs (two threads). That overhead is bigger than the entire kernel
on small groups (13 µs density), so concurrency on the GPU is overwhelmed
by serialization in the host driver / cuBLAS / OMP runtime.

### Likely sources of the host-side contention (not individually isolated)

1. **CUDA driver locks**. Stream-ordered alloc (`cudaMallocAsync`),
   stream creation, texture creation, and submission-queue management
   all take global driver locks. With two host threads both submitting
   work at ~kHz launch rates, contention dominates.
2. **cuBLAS internal lazy init**. Each thread-local handle has to
   initialize routing tables, scratch pools, and internal streams on
   first use of each kernel size; running two of them in parallel
   doubles the lazy-init cost. Past the first few iterations this should
   amortize, but the steady-state launch rate still bottoms out higher.
3. **OMP thread overhead at this granularity**. Per-step GPU work is
   ~570 µs; spinning up 6 worker threads (4 idle CPU + 2 GPU) every
   step adds visible overhead because the work per thread is so small.
   When combined with the `LIO_OVERLAP_INT3LU_G2G` nested OMP team
   (sections × 2 + g2g × 6 + BLAS × 4 = 12 concurrent threads on 8
   physical cores), the system over-subscribes and the overlap path
   hangs / starves.

## Why kernel concurrency didn't make up for it

Even when two workers do successfully launch kernels concurrently, the
per-SM L1/TEX cache is shared and the density kernel saturates it at ~94%
on the *large* groups. The small groups that benefit from concurrency
(gridX 59–367, ~408 ms total in the run) are only a ~9% pool of total GPU
time. With realistic capture rates the win is bounded under 5% wall —
which the host overhead overwhelms by a factor of 3.

## What stays in the tree

- `--default-stream per-thread` compile flag (no regression on single-thread,
  enables future experimentation if a finer-grained variant is tried).
- `LIO_GPU_THREADS` env var, default 1 (no behavior change). The variable
  is *opt-in*; setting it > 1 will be slower until the host contention is
  understood.
- Thread-local cuBLAS handle (necessary for any future multi-thread driver
  experiment, harmless for single-thread).

## When to reconsider

- **Single-thread + N explicit streams** (round-robin per group). Removes
  cuBLAS handle and OMP contention. Per-thread default stream is *not* used;
  each kernel takes an explicit stream argument. Expected ceiling: ~5% wall
  on TDDFT, ~0% on SCF. Cost: ~100 LOC of stream parameters threaded
  through every `<<<>>>` launch in `iteration.cu`. **Not worth doing before
  the inter-step gap [[td-inter-step-gap]] is attacked — that pool is ~28%
  wall.**
- A different GPU. Hopper / Blackwell have larger L2 and different SM
  scheduling; the small-group concurrency story may improve.
- A newer CUDA version where stream-ordered alloc and cuBLAS init have
  lower locks (CUDA 12.4+ per release notes, untested here).

## Cross-references

- [[td-inter-step-gap]] — 1.3s pool on the same workload, much bigger lever
- [[path4-cuda-graphs-2026-05-20]] — earlier rejected attempt at the same
  per-step launch-overhead pool, also net-neutral
