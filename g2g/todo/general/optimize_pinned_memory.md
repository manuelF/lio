# Optimization: Pinned (Page-Locked) Host Memory for All H2D/D2H Transfers

## Summary
`cudaMemcpyAsync` from **pageable** host memory is NOT asynchronous. The CUDA driver
internally stages the copy through a pinned bounce buffer, serializing the transfer
on the CPU side before DMA can proceed. Only transfers from **pinned** (page-locked)
memory (`cudaMallocHost` / `cudaHostAlloc`) are truly DMA-driven and async.

LIO uses `copy_submatrix_async` for results readback (energy, forces, rmm_output)
but the underlying `HostMatrix` is allocated with `new T[]` (pageable) by default.
This means all these "async" calls actually block until the driver staging is complete.

See `matrix.cpp:41–43`:
```cpp
if (pinned) {
  cudaError_t error_status = cudaMallocHost((void**)&this->data, this->bytes());
```
The `Pinned` flag exists but is not used for iteration buffers.

## Audit: Buffers That Should Be Pinned

### In `iteration.cu` (PointGroupGPU::solve_closed):

| Buffer | Current | Size (typical) | Transfer |
|---|---|---|---|
| `rmm_input_cpu` | `new float[]` (pageable) | M²×4 B ≈ 3.6 KB | H2D per group |
| `energy_host` | `new float[]` (pageable) | P×4 B ≈ 2 KB | D2H per group |
| `forces_host` | `new float[]` (pageable) | N_atoms×3×4 B ≈ varies | D2H per group |
| `rmm_output_host` | `new float[]` (pageable) | M(M+1)/2×4 B ≈ 1.8 KB | D2H per group |
| `point_weights_cpu` | `new float[]` (pageable) | P×4 B ≈ 2 KB | H2D per group |
| `function_values` | `new float[]` (pageable) | M×P×4 B ≈ 60 KB | H2D staging |

All of these are transferred every group, every SCF step. With 100 groups, that's
100 pageable H2D + 100 D2H per step — all staged through bounce buffers.

### In `partition.cpp`:
- The `Partition::fort_forces_ms` matrices are `HostMatrix<double>` used for force
  accumulation. Not directly transferred to GPU but written by CPU while GPU computes.

## Proposal

### Approach A — Mark Iteration Buffers as Pinned (Simplest)

In `PointGroupGPU::initialize()` or as member declarations:
```cpp
// partition.h:
HostMatrix<scalar_type> rmm_input_cpu{HostMatrix<scalar_type>::Pinned};
HostMatrix<scalar_type> energy_host  {HostMatrix<scalar_type>::Pinned};
HostMatrix<scalar_type> forces_host  {HostMatrix<scalar_type>::Pinned};
HostMatrix<scalar_type> rmm_output_host{HostMatrix<scalar_type>::Pinned};
```

This is the minimal change. `HostMatrix::Pinned` already exists and maps to
`cudaMallocHost`.

**Caveat**: `cudaMallocHost` is more expensive than `malloc` for small sizes. These
buffers are allocated once at group creation and freed at group destruction, so the
amortized cost is negligible.

### Approach B — Pinned Memory Pool (Advanced)

Allocate a single large pinned buffer at startup (e.g., 32 MB) and sub-allocate
from it for all iteration buffers. This avoids the per-buffer `cudaMallocHost`
overhead:
```cpp
cudaHostAlloc(&pinned_pool, 32 * 1024 * 1024, cudaHostAllocDefault);
// Then use bump allocator for rmm_input_cpu, energy_host, etc.
```
This also enables `cudaHostGetDevicePointer` for zero-copy access on integrated GPUs
(not relevant for GTX 1080 discrete GPU, but future-proof).

### Approach C — Align with GPU RMM Gather (Best Long-Term)

Once `optimize_rmm_gather_gpu.md` is implemented, `rmm_input_cpu` becomes obsolete
(the gather happens on GPU directly). The list of buffers requiring pinned memory
shrinks to just `energy_host`, `forces_host`, and `rmm_output_host` — all small.

## Impact

### Transfer Bandwidth Without Pinned (current)
PCIe 3.0 ×16: 16 GB/s peak. Driver-staged copies for small buffers: effectively
4–6 GB/s due to CPU memcpy overhead in staging.

### Transfer Bandwidth With Pinned
DMA-direct: 12–14 GB/s effective, and fully async (CPU is free during transfer).

For a group with 3.6 KB (rmm_input_cpu) + 2 KB (energy) + 1.8 KB (rmm_output):
- Without pinned: ~7.4 KB / 5 GB/s ≈ **1.5 µs per group** (blocking)
- With pinned: ~7.4 KB / 13 GB/s ≈ **0.57 µs per group** (async, overlapped)

For 100 groups: 150 µs → 57 µs of transfer time, plus 93 µs of CPU unblocking.
**Effective saving: ~0.1–0.5 ms per SCF step** for typical systems.

In combination with `async_execution.md` (double buffering), pinned memory is
**mandatory** to achieve true CPU-GPU overlap. The async transfers only overlap
if the host memory is pinned.

## Difficulty Assessment
**Low** (Approach A)

- Files: `g2g/partition.h` (add `Pinned` flag to member declarations) and
  `g2g/cuda/iteration.cu` (ensure `resize` calls don't switch to non-pinned).
- Correctness: **Zero risk** — pinned memory is behaviorally identical to pageable.
- The `HostMatrix::Pinned` enum value already exists; just needs to be used.

## Files to Modify
- `g2g/partition.h`: Change `HostMatrix<scalar_type>` member declarations for
  `rmm_input_cpu`, `energy_host`, `forces_host`, `rmm_output_host` to add `Pinned`.
- `g2g/cuda/iteration.cu`: Verify `resize()` calls preserve the pinned flag
  (currently `resize` calls `dealloc_data` + `alloc_data`, which re-checks `pinned`
  member — so it's safe).

## Prerequisite Note
This optimization is a **prerequisite** for `async_execution.md` Phase 3
(double buffering). Without pinned memory, double buffering provides no benefit
since transfers still block on the CPU.

## Estimations
- Transfer time reduction: **2–4× faster per-transfer** (pageable staging eliminated).
- CPU unblocking: frees CPU cycles during D2H results readback.
- End-to-end: **3–8% overall speedup** standalone; **enables 10–20% additional**
  when combined with double-buffering pipeline.
- Implementation: **2–4 hours** for Approach A.
