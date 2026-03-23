# Optimization: Async Execution and CPU-GPU Overlap

## Summary (Corrected — Code Audit Findings)

The original description overstated the problem. `iteration.cu` **already uses**
`cudaMemcpy2DToArrayAsync` for RMM copies (lines ~243, 470, 699, 806) and
`copy_submatrix_async` for result readbacks (lines ~382, 507, 545). Timers use
`timers.xxx.start()` / `timers.xxx.pause()` — not `start_and_sync()`.

**What is actually blocking and serializating:**

1. **`get_rmm_input()` CPU loop (most expensive)**: Runs entirely on CPU after all
   GPU kernels for one kernel phase complete. It gathers/scatters the global RMM
   (density matrix) from Fortran packed storage into a local group-sized matrix,
   then initiates `cudaMemcpy2DToArrayAsync`. This CPU work happens while the GPU
   could be doing other work. Solution: move this to the GPU — see
   `optimize_rmm_gather_gpu.md`.

2. **Single-stream execution**: All kernels execute on the default stream (stream 0).
   No CPU-GPU overlap; no concurrent group processing. While one group's kernels run,
   the CPU prepares nothing useful.

3. **Pinned memory not consistently used**: `HostMatrix` defaults to `NonPinned` (uses
   `new T[...]`). Async copies from pageable host memory are internally staged through
   a pinned buffer by the CUDA driver, adding latency. See `optimize_pinned_memory.md`.

4. **Timer overhead**: `Timer::start()` and `Timer::pause()` insert `cudaEventRecord`
   on the default stream, which is fine for profiling but adds driver calls per group.
   During production runs, this overhead accumulates across hundreds of groups.

## Proposal

### Phase 1 — Pinned Memory + True Async (Low Risk)
- Allocate `rmm_input_cpu`, `forces_host`, `energy_host`, `rmm_output_host` as
  `HostMatrix<T>(Pinned)` so async copies are DMA-direct without driver staging.
- This alone makes the existing `copy_submatrix_async` calls actually asynchronous.
- Files: `g2g/cuda/iteration.cu` — change allocation flags.

### Phase 2 — RMM Gather on GPU (Highest Impact)
- Port `get_rmm_input()` to a CUDA kernel `gpu_gather_rmm<<<>>>` that reads the global
  RMM from a GPU buffer and writes the local group-sized subset to `rmm_cuArray`.
- Eliminates the D2H → CPU-shuffle → H2D round-trip entirely.
- See `optimize_rmm_gather_gpu.md` for full specification.

### Phase 3 — Double Buffering Pipeline (High Impact, High Complexity)
- Create two `Workspace` objects per stream: while GPU executes group N,
  CPU+copy-engine prepares group N+1.
- Requires restructuring `Partition::solve` to submit and synchronize with a lookahead
  of 1 group.

### Phase 4 — Timer Bypass in Production Mode (Easy)
- Wrap timer calls in `#ifdef LIO_PROFILE` or check a runtime flag.
- Removes `cudaEventRecord` overhead on the hot path.

## Impact
- Phase 1 (pinned): **5–10% latency reduction** for memory-transfer-bound groups.
- Phase 2 (GPU RMM gather): **20–40% overall speedup** (removes dominant CPU stall).
- Phase 3 (double buffering): **10–20% additional** on top of Phase 2.
- Phase 4 (timers): **1–3%** micro-optimization.

## Difficulty Assessment
- Phase 1: **Low** (change allocation flags)
- Phase 2: **Medium** (new kernel, index mapping, correctness-critical)
- Phase 3: **High** (restructure solve loop, synchronization)
- Phase 4: **Trivial**

## Files to Modify
- `g2g/cuda/iteration.cu`: All phases.
- `g2g/matrix.cpp`: Phase 1 — default allocation flag for iteration buffers.
- `g2g/partition.h` / `g2g/partition.cpp`: Phase 3 — workspace double-buffering.
- `g2g/cuda/kernels/` (new file): Phase 2 — `rmm_gather.h`.

## Correctness Risk
- Phase 1: **None** — behavior-identical, just faster.
- Phase 2: **High** — Fortran packed indexing must be reproduced exactly. The
  existing `get_rmm_input` has upper/lower triangle logic that must be ported correctly.
  Run `agua`, `fosfato`, `Fe3H2O6` tests after every change.
- Phase 3: **High** — stream synchronization errors cause data corruption.

## Estimations
- Combined Phase 1+2: **25–45% overall runtime improvement**.
- Combined Phase 1+2+3: **35–60%** for large systems with many groups.
