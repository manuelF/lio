# CUDA Optimization Guidebook

Reference guide sorted by **Performance Criticality** (what to fix first),
categorized by **Optimization Dimension** (Memory, Compute, Host/Device, Tooling).

---

## 🔴 TIER 1: CRITICAL (Fundamentals & Major Bottlenecks)

> Skip these and your GPU will perform like a slow CPU.
> Fix these before trying anything else.

---

### 🧠 Memory & Data Movement

#### Coalesce global memory access

Threads in a warp (32 threads) should access **contiguous** global memory addresses.
When thread `i` accesses `data[i]`, the hardware merges all 32 reads into a single
128-byte transaction. When threads access non-contiguous or strided addresses, each
access becomes a separate transaction — up to 32× more bandwidth consumed.

```cuda
// Bad: column-major access — each thread strides by `rows` elements
float val = matrix[threadIdx.x * rows + blockIdx.x];  // stride = rows

// Good: row-major access — contiguous warp access
float val = matrix[blockIdx.x * cols + threadIdx.x];  // stride = 1
```

**Diagnosis:** In Nsight Compute, check `l2_global_load_bytes` vs `dram_read_bytes`.
A high ratio (>4 sectors/request) indicates poor coalescing.

---

#### Use Structure of Arrays (SoA) instead of Array of Structures (AoS)

AoS packs heterogeneous fields together, causing strided access when a kernel reads
only one field. SoA separates fields into contiguous arrays, enabling coalesced access.

```cuda
// Bad: AoS — reading .x strides over the entire struct
struct Particle { float x, y, z, mass; };
Particle particles[N];
float xi = particles[threadIdx.x].x;  // stride = sizeof(Particle) = 16 bytes

// Good: SoA — reading x[] is fully coalesced
float x[N], y[N], z[N], mass[N];
float xi = x[threadIdx.x];  // stride = 4 bytes, perfectly coalesced
```

*In LIO:* `CudaMatrix<vec_type4>` stores (x,y,z,w) per element — acceptable when all
four components are read together. Avoid reading a single component from a `vec_type4`
array in a warp-strided pattern.

---

#### Keep data on the GPU as long as possible

Every PCIe round-trip (H→D→H) is expensive: ~10–20 GB/s on PCIe 3.0 vs ~700 GB/s
GPU memory bandwidth. Avoid downloading intermediate results to the CPU and uploading
them again. Structure the pipeline so GPU buffers persist across kernel launches.

```
Bad:   [GPU kernel A] → D→H copy → [CPU work] → H→D copy → [GPU kernel B]
Good:  [GPU kernel A] → [GPU kernel B]  (CPU work fused into kernels or eliminated)
```

*In LIO:* `rmm_output_gpu` → `rmm_output_host` → `add_rmm_output()` is a necessary
scatter step (because the output index mapping is complex), but the function values
(`function_values_transposed`) are correctly kept on GPU via `inGlobal` caching.

---

### ⚡ Compute & Execution

#### Minimize warp divergence

All 32 threads in a warp execute the **same instruction** at the same time (SIMT).
When threads take different branches (`if/else`), both paths are serialized — the
inactive threads stall. Total cost = sum of all branch path times, not the max.

```cuda
// Bad: threads diverge based on threadIdx
if (threadIdx.x % 2 == 0) {
    // only even threads work — odd threads stall
} else {
    // only odd threads work — even threads stall
}

// Better: rearrange data so the branch is uniform across the warp
// Or use predicated execution for small branches
val = (condition) ? a : b;  // compiler may emit a predicated select
```

**When it's unavoidable:** Triangular loops (`for j <= i`) inherently diverge.
Minimize work in divergent paths; consider padding data to make blocks uniform.

*In LIO:* `gpu_compute_density` has a triangular loop condition — `warp_execution_efficiency`
is ~45% as a result. This is a known, structural limitation.

---

#### Set thread block sizes to multiples of 32

A warp is 32 threads. Blocks that are not multiples of 32 waste hardware slots: a block
of 33 threads requires 2 warps (64 slots), leaving 31 threads idle. Block sizes of 64,
128, 256, or 512 are typical; 256 is a common default.

```cuda
// Bad: wastes warp slots
dim3 block(100);  // 4 warps, 28 idle threads in last warp

// Good: exactly 3 full warps
dim3 block(96);  // 3 warps, 0 idle threads
```

*In LIO:* `DENSITY_BLOCK_SIZE = 64`, `DENSITY_ACCUM_BLOCK_SIZE = 256`,
`WEIGHT_BLOCK_SIZE`, `FUNCTIONS_BLOCK_SIZE` — all defined in `common.h` as multiples
of 16 (must remain so).

---

#### Maximize occupancy

Occupancy = active warps / maximum warps per SM. Higher occupancy hides memory latency
by keeping the SM busy with other warps while waiting for data. Two main limiters:

- **Register usage:** Each SM has 65,536 registers (SM 6.1). A kernel using 32 registers
  per thread allows 2048 threads/SM = 100% occupancy. At 64 registers: 1024 threads = 50%.
- **Shared memory:** Each SM has 48–96 KB. Large shared memory allocations reduce the
  number of concurrent blocks.

Check with: `nvprof --metrics achieved_occupancy` or Nsight Compute → Occupancy section.

```cuda
// Hint the compiler to target specific register/thread counts:
__launch_bounds__(256, 4)  // maxThreadsPerBlock=256, minBlocksPerSM=4
__global__ void my_kernel(...) { ... }
```

*In LIO measured (SM 6.1, float):*
| Kernel | Regs | smem | Occupancy |
|---|---|---|---|
| gpu_compute_density (LDA) | 32 | 256 B | 100% |
| gpu_compute_density (GGA) | 56 | 2560 B | 56% |
| gpu_compute_density_opened (GGA) | 93 | 2560 B | 34% ← bottleneck |

---

#### Prefer `float` over `double`

On consumer GPUs (Pascal, Turing, Ampere consumer), FP64 throughput is typically
1/32 of FP32. `double` arithmetic is 32× slower and uses more registers.

Use `double` only where algorithm correctness requires it (e.g., energy accumulation,
density matrix elements). Cast to `float` for GPU-side computation where precision is
sufficient.

```cuda
// Bad: unnecessary double in GPU kernel
double result = a * b + c;

// Good: float computation, double accumulation only at reduction stage
float result = a * b + c;
// ... accumulate many floats into a double at the end
```

*In LIO:* `FULL_DOUBLE` macro controls precision. Default is hybrid (float kernels,
double accumulation). Full double (`make precision=1`) is ~32× slower on GTX 1080.

---

### 🔄 Host-Device & Concurrency

#### Minimize host-device synchronization

`cudaDeviceSynchronize()` blocks the host until ALL GPU work completes. Use it only
when strictly necessary (e.g., before reading GPU-side results on the CPU, or at
program exit for error checking).

Prefer stream-level sync (`cudaStreamSynchronize(stream)`) which only blocks until
that stream's work completes, allowing other streams to continue.

```cuda
// Bad: blocks everything
kernel_A<<<...>>>(d_a);
cudaDeviceSynchronize();  // stalls host until A finishes
kernel_B<<<...>>>(d_b);   // CPU was idle during A's execution

// Better: overlap using streams
kernel_A<<<..., stream_1>>>(d_a);
kernel_B<<<..., stream_2>>>(d_b);   // launches immediately
cudaStreamSynchronize(stream_1);    // only blocks for A
```

---

#### Minimize `cudaDeviceSynchronize()` calls

Every call stalls the host process until the GPU finishes ALL pending work across ALL
streams. This creates a serialization point that prevents any CPU/GPU overlap.

Common hidden sync sources:
- `cudaMalloc` / `cudaFree` (may implicitly sync)
- `cudaMemcpy` (synchronous version — use `cudaMemcpyAsync` instead)
- `cudaError_t` error checks with `cudaGetLastError()` after kernel launches (safe;
  does not sync)

```cuda
// Error checking without sync (preferred):
kernel<<<grid, block>>>(args);
cudaError_t err = cudaGetLastError();  // does NOT synchronize
if (err != cudaSuccess) { /* handle */ }

// Only sync when you actually need the result:
cudaStreamSynchronize(stream);
read_results_on_cpu();
```

---

### 🛠 Tooling & Architecture

#### Profile with Nsight Compute

Determines if a kernel is **memory-bound** or **compute-bound** and why.

```bash
# Quick metrics summary to stdout
ncu --set basic ./program

# Full profile saved for GUI analysis
ncu -o report --kernel-name "my_kernel" ./program

# Key metric sets:
ncu --set memory    # memory hierarchy analysis
ncu --set roofline  # arithmetic intensity vs peak bandwidth/FLOPS
ncu --set full      # everything (slow)
```

Key metrics to watch:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — overall SM utilization
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / requests — sectors/request (coalescing)
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` — actual FLOP count
- `smsp__warp_issue_stall_*` — where warps are stalling (memory? sync? dependency?)

**Note:** Nsight Compute requires SM 7.0+. On SM 6.1 (GTX 1080), use `nvprof` instead:
```bash
nvprof --metrics achieved_occupancy,stall_memory_dependency ./program
nvprof -o profile.nvvp ./program   # save for visual profiler
```

---

#### Use Nsight Systems

Identifies gaps in the GPU execution timeline: kernel launch gaps, CPU/GPU serialization,
unnecessary synchronization, or PCIe transfer bottlenecks.

```bash
nsys profile -o report ./program
nsys stats report.nsys-rep --report cuda_gpu_kern_sum   # kernel time summary
nsys stats report.nsys-rep --report cuda_api_sum        # API call summary
```

Look for:
- **Gaps between kernels**: launch overhead or synchronization
- **Long H→D or D→H transfers**: data being unnecessarily round-tripped
- **CPU time between GPU operations**: overlapping opportunity

---

#### Apply roofline analysis

The roofline model plots a kernel's **arithmetic intensity** (FLOPs / bytes) against
hardware peak limits to show whether it's bounded by memory bandwidth or compute.

```
FLOP/s
  |                        /‾‾‾‾ Compute ceiling (peak TFLOPS)
  |                   ____/
  |              ____/  Compute-bound kernels live here
  |         ____/
  |    ____/ Memory-bound kernels live here (below the slope)
  |___/__________________________________________ Arithmetic intensity (FLOP/byte)
```

If your kernel is memory-bound: optimize memory access patterns (coalescing, caching).
If compute-bound: reduce arithmetic, use faster math, or increase parallelism.

```bash
ncu --set roofline -o report ./program
# Open report.ncu-rep in Nsight Compute GUI → "Roofline" section
```

---

#### Check every CUDA API call

Silent CUDA errors corrupt results and are nearly impossible to debug later.
Use a macro that checks every API call:

```cuda
#define CUDA_CHECK(call)                                              \
  do {                                                               \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
              __FILE__, __LINE__, cudaGetErrorString(err));          \
      exit(1);                                                       \
    }                                                                \
  } while (0)

CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
```

For kernel launches (which don't return an error directly):
```cuda
my_kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());   // checks launch parameters
// cudaStreamSynchronize() or cudaDeviceSynchronize() is needed to catch runtime errors
```

*In LIO:* `cudaAssertNoError(label)` is the existing macro in `cuda_extra.h`.

---

#### Avoid `printf` in performance kernels

`printf` in device code requires a per-SM circular buffer, introduces global memory
traffic, and serializes thread execution. Even a single `printf` path in a hot kernel
can reduce throughput by 10–100×.

```cuda
// Bad in performance kernel:
__global__ void hot_kernel(float* data) {
    float val = compute(data[threadIdx.x]);
    printf("thread %d: val=%f\n", threadIdx.x, val);  // kills performance
}

// OK for debugging — gate with a compile-time flag:
#ifdef DEBUG_PRINTS
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("block 0 entry: n=%d\n", n);
#endif
```

---

## 🟡 TIER 2: HIGH IMPACT (Architecture & Algorithmic Tuning)

> Once the fundamentals are right, these provide the most significant speedups.

---

### 🧠 Memory & Data Movement

#### Utilize shared memory as a user-managed cache

Shared memory (~48 KB per SM, ~1–2 cycles latency) is orders of magnitude faster than
global memory (~300–500 cycles). Use it to cache data that multiple threads in a block
reuse.

```cuda
__global__ void with_shared(float* in, float* out, int n) {
    __shared__ float tile[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Load once from slow global memory
    tile[threadIdx.x] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    // All subsequent accesses use fast shared memory
    float result = tile[threadIdx.x] + tile[(threadIdx.x + 1) % BLOCK_SIZE];
    if (idx < n) out[idx] = result;
}
```

**Pattern:** Load tile from global → `__syncthreads()` → compute from shared → store.

*In LIO:* `gpu_compute_density` caches rows of `function_values_transposed` in shared
memory (`fj_sh[]`) to avoid repeated global reads during the bj-loop.

---

#### Avoid shared memory bank conflicts

Shared memory is divided into 32 banks (4-byte interleaved). When multiple threads in
a warp access the **same bank** simultaneously (but different addresses), accesses are
serialized — a "bank conflict."

```cuda
// Bad: all threads access bank 0 (stride = 32 words = one full bank cycle)
__shared__ float s[32];
float val = s[threadIdx.x * 32];  // all hit bank 0 → 32-way conflict

// Good: each thread hits a different bank
float val = s[threadIdx.x];       // thread i → bank (i % 32)

// Padding trick to break stride-32 conflicts in 2D tiles:
__shared__ float tile[32][33];    // +1 column shifts bank per row
```

---

#### Use pinned (page-locked) host memory

Standard `malloc` memory can be paged out by the OS. CUDA must stage async copies
through an internal pinned buffer, halving effective bandwidth. Pinned memory allows
the DMA engine to transfer directly, doubling throughput.

```cuda
float* h_data;
cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);  // pinned
// ... use h_data normally on CPU ...
cudaMemcpyAsync(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream);
cudaFreeHost(h_data);
```

**Warning:** Pinned memory is a limited resource. Over-pinning causes the OS to page
other allocations, degrading overall system performance. Pin only transfer-critical buffers.

*In LIO:* `rmm_input_cpu_cache` and `rmm_output_host` use `HostMatrix<T>::Pinned` for
exactly this reason.

---

#### Avoid register spilling to local memory

When a kernel uses more registers than the per-thread limit (65,536 / block_size),
the excess "spills" to **local memory** — which is physically global memory, but
thread-private. Local memory accesses are cached in L1 but have the same latency as
global memory when they miss.

Signs of spilling:
- `nvprof --metrics local_load,local_store` shows non-zero counts
- Nsight Compute → Source → high "Local Memory" traffic

Mitigations:
- Reduce the number of live variables (fewer temporaries, shorter live ranges)
- Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` to guide the compiler
- Split the kernel into smaller phases

*In LIO:* `gpu_compute_density_opened` (open-shell GGA) uses 93 registers → 34%
occupancy on SM 6.1. Target: reduce to ~56 registers by replacing with 2 calls to
the closed-shell kernel. See `g2g/todo/gpu/optimize_open_shell_registers.md`.

---

#### Avoid large local arrays in kernels

Arrays declared inside a kernel (`float arr[N]`) with runtime-unknown size or large
static size are placed in local memory (global memory, thread-private). Each access
incurs full global memory latency.

```cuda
// Bad: large stack array → likely in local memory
__global__ void bad_kernel() {
    float workspace[1024];  // 4 KB per thread → local memory
    ...
}

// Good: use shared memory for data shared within a block
__global__ void good_kernel() {
    __shared__ float workspace[1024];  // shared across all threads in block
    ...
}
```

---

#### Utilize constant memory

Constant memory (64 KB total, cached) is optimal for **read-only data broadcast to
all threads in a warp** (e.g., simulation parameters, atom positions). When all threads
read the same address, it's served from a single cache read.

```cuda
__constant__ float gpu_atom_positions[MAX_ATOMS * 3];

// Upload once:
cudaMemcpyToSymbol(gpu_atom_positions, host_positions, size);

// Access in kernel (all threads read same address → 1 memory transaction):
float pos_x = gpu_atom_positions[atom_id * 3];
```

*In LIO:* `gpu_atom_positions`, `gpu_atoms`, `gpu_Iexch`, `gpu_normalization_factor`
are all constant-memory symbols declared in `gpu_variables.h`.

---

#### Use vectorized loads (`float4`, `int2`)

A single `float4` load fetches 16 bytes in one instruction instead of four 4-byte loads.
This reduces instruction count and better saturates the memory bus.

```cuda
// Bad: 4 separate 4-byte loads
float a = data[4*i], b = data[4*i+1], c = data[4*i+2], d = data[4*i+3];

// Good: 1 vectorized 16-byte load
float4 v = reinterpret_cast<float4*>(data)[i];
float a = v.x, b = v.y, c = v.z, d = v.w;
```

Requirements: pointer must be 16-byte aligned, and `N % 4 == 0` (or handle the tail).

*In LIO:* `gradient_values`, `dxyz_gpu`, `dd1_gpu`, `dd2_gpu` are all
`CudaMatrix<vec_type4>` — they use `float4` storage for exactly this reason.

---

### ⚡ Compute & Execution

#### Explore mixed-precision (FP16/BF16)

FP16/BF16 offer 2× arithmetic throughput over FP32 on Volta+ hardware, and 16× on
Tensor Core operations. Use for non-critical intermediate computations.

```cuda
#include <cuda_fp16.h>
__half a = __float2half(1.5f);
__half b = __float2half(2.0f);
__half c = __hadd(a, b);  // FP16 add
float result = __half2float(c);
```

*Note:* FP16 has limited dynamic range (max ~65,504). Use where values are bounded.
*In LIO:* Not currently used. Potential for gradient accumulation if precision is verified.

---

#### Utilize Tensor Cores

Tensor Cores (Volta+) perform 4×4 matrix multiply-accumulate in a single instruction,
providing massive throughput for matrix operations (GEMM). Access via CUTLASS or cuBLAS.

```cuda
// Via cuBLAS (handles Tensor Core selection automatically):
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
cublasSgemm(handle, ...);  // uses Tensor Cores if shapes are compatible
```

*In LIO:* `make cuda=2` enables CUBLAS. Worth evaluating for the density matrix
contraction step (`gpu_update_rmm`) if reformulated as a GEMM.

---

#### Implement grid-stride loops

Instead of one thread per element, each thread handles multiple elements with a stride
of the total grid size. This handles arrays larger than the grid, improves instruction
reuse, and makes kernel launch parameters flexible.

```cuda
// Bad: assumes exactly N threads
__global__ void kernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = compute(data[i]);  // fails if i >= N or grid too small
}

// Good: grid-stride loop
__global__ void kernel(float* data, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * blockDim.x) {
        data[i] = compute(data[i]);
    }
}
```

---

#### Employ warp-level primitives (`__shfl_sync`)

Warp shuffle instructions transfer values between threads **within a warp** using
registers only — no shared memory, no synchronization barriers. Ideal for reductions,
broadcasts, and prefix scans within a warp.

```cuda
// Warp reduction using shuffle (replaces volatile shared memory tree):
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;  // lane 0 holds the sum
}
```

**Performance:** Eliminates 1 `__syncthreads()` and all shared memory traffic for
intra-warp reduction.

*In LIO:* Already applied in `energy.h` and `energy_open.h`. See
`g2g/todo/gpu/optimize_warp_shuffle.md`.

---

#### Implement tree-based reductions

Naive summation (one thread accumulates all values) is O(N). A parallel tree reduction
is O(log N) steps with full warp utilization.

```cuda
// Parallel tree reduction in shared memory:
__shared__ float sdata[BLOCK_SIZE];
sdata[tid] = partial_sum;
__syncthreads();

for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
// Finish with warp shuffle for last 32 elements (no syncthreads needed)
if (tid < 32) sdata[tid] = warpReduceSum(sdata[tid]);
```

---

#### Apply `#pragma unroll`

Instructs the compiler to unroll a loop, eliminating branch overhead and exposing more
instruction-level parallelism. Most effective on short, fixed-count loops.

```cuda
// Let compiler choose unroll factor:
#pragma unroll
for (int i = 0; i < 4; i++) acc += data[i];

// Explicit unroll count:
#pragma unroll 4
for (int i = 0; i < N; i++) acc += a[i] * b[i];  // unroll 4 iterations

// Disable unrolling (useful when register pressure is a concern):
#pragma unroll 1
for (int i = 0; i < N; i++) acc += a[i];
```

*In LIO:* Used in `energy_open.h` inner bj-loop: `#pragma unroll 4`.

---

#### Use fast math intrinsics

`--use_fast_math` (or `-ftz=true -prec-div=false -prec-sqrt=false`) maps standard
math functions to faster approximations. Individual intrinsics give explicit control:

| Standard | Fast intrinsic | Speed | Precision |
|---|---|---|---|
| `sinf(x)` | `__sinf(x)` | ~4× faster | ~5 ULP vs 1 ULP |
| `expf(x)` | `__expf(x)` | ~4× faster | ~2 ULP |
| `a / b` | `__fdividef(a, b)` | ~2× faster | ~2 ULP |
| `sqrtf(x)` | `__fsqrt_rn(x)` | same speed | exact |
| `1/sqrtf(x)` | `__frsqrt_rn(x)` | ~2× faster | ~2 ULP |

*In LIO:* `--use_fast_math` is already in `g2g/Makefile.cuda` (`-use_fast_math`).

---

#### Reduce global atomic contention

Global atomics (`atomicAdd` on a global pointer) serialize when multiple threads target
the same address. Reduce contention by first aggregating in shared memory, then doing
one global atomic per block.

```cuda
// Bad: N threads all atomicAdd to same global location
atomicAdd(global_sum, local_val);  // contention = blockDim.x way

// Good: reduce in shared first, one global atomic per block
__shared__ float sdata[BLOCK_SIZE];
sdata[tid] = local_val;
__syncthreads();
// ... tree reduction ...
if (tid == 0) atomicAdd(global_sum, sdata[0]);  // 1 atomic per block
```

---

#### Implement thread coarsening

In memory-bound kernels, launching more threads than necessary can waste bandwidth
(each thread reads its own cache line). Coarsening assigns more work per thread,
allowing reuse of loaded data across iterations.

```cuda
// Fine-grained: each thread handles 1 element
int i = blockIdx.x * blockDim.x + threadIdx.x;
out[i] = process(in[i]);

// Coarsened: each thread handles COARSE elements from the same cache line
int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE;
for (int k = 0; k < COARSE; k++)
    out[base + k] = process(in[base + k]);
```

---

### 🔄 Host-Device & Concurrency

#### Overlap data transfers with execution

Use multiple streams to pipeline H→D transfers, kernel execution, and D→H transfers
concurrently. Requires pinned host memory and async API calls.

```
Stream 1: |--H→D--|  |--kernel--|  |--D→H--|
Stream 2:          |--H→D--|  |--kernel--|  |--D→H--|
Timeline:  [========== overlap ===========]
```

```cuda
for (int i = 0; i < num_chunks; i++) {
    cudaMemcpyAsync(d_in[i], h_in[i], chunk_size, cudaMemcpyHostToDevice, stream[i]);
    kernel<<<grid, block, 0, stream[i]>>>(d_in[i], d_out[i]);
    cudaMemcpyAsync(h_out[i], d_out[i], chunk_size, cudaMemcpyDeviceToHost, stream[i]);
}
```

*In LIO:* `transpose_stream_1/2` overlap with `get_rmm_input()` CPU work.
See `g2g/CLAUDE.md` → "Stream fix" (commit 21758bcb).

---

#### Use `cudaMemcpyAsync`

The synchronous `cudaMemcpy` blocks the CPU until the transfer completes. The async
variant returns immediately, allowing the CPU to continue work or launch more kernels.

```cuda
// Synchronous — CPU blocks until copy finishes:
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);

// Asynchronous — CPU continues immediately (requires pinned src):
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
// CPU can do other work here...
cudaStreamSynchronize(stream);  // wait only when result is needed
```

---

#### Use CUDA Graphs

CUDA Graphs capture a sequence of kernels and memory operations into a reusable
execution descriptor. Avoids per-launch CPU overhead (~5–10 µs per kernel launch),
which matters when launching many small kernels.

```cuda
// Capture a graph:
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_A<<<grid, block, 0, stream>>>(args);
kernel_B<<<grid, block, 0, stream>>>(args);
cudaStreamEndCapture(stream, &graph);

// Instantiate and launch (cheap — ~microseconds):
cudaGraphExec_t exec;
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(exec, stream);  // launches both kernels with minimal overhead
```

---

#### Prefetch unified memory data

When using `cudaMallocManaged`, page faults on first access cause stalls. Prefetch
data to the target device before it is needed:

```cuda
cudaMallocManaged(&data, size);
// ... initialize data on CPU ...

// Prefetch to GPU before the kernel runs:
cudaMemPrefetchAsync(data, size, device_id, stream);
kernel<<<grid, block, 0, stream>>>(data);
```

---

### 🛠 Tooling & Architecture

#### Target specific architectures

Compile for the exact GPU on the target machine for best code generation.
Always include both PTX (forward compatibility) and cubin (direct execution) entries.

```makefile
# Auto-detect from nvidia-smi:
DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap \
                 --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
GENCODE_FLAGS := -gencode arch=compute_$(DETECTED_SM),code=compute_$(DETECTED_SM)
GENCODE_FLAGS += -gencode arch=compute_$(DETECTED_SM),code=sm_$(DETECTED_SM)
```

`code=compute_XX` = PTX (JIT-compiled at runtime for newer GPUs)
`code=sm_XX` = cubin (native code, fastest on that exact GPU)

*In LIO:* This exact pattern is in `g2g/Makefile.cuda`.

---

#### Use `cudaEventRecord` for precise timing

CPU timers include OS scheduling jitter. GPU event timers measure exact GPU-side elapsed
time, unaffected by CPU load.

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
```

---

#### Regularly update the CUDA Toolkit

New CUDA versions include:
- Improved compiler optimization passes (register allocation, instruction scheduling)
- New intrinsics and hardware features support
- Bug fixes in `nvcc` code generation
- Improved Nsight profiler support

Check release notes for performance regressions on your architecture before upgrading.

---

## 🟢 TIER 3: MEDIUM / MICRO (Fine-Tuning & Squeezing)

> Advanced techniques to extract the last 5–15% from the hardware.

---

### 🧠 Memory & Data Movement

#### Leverage L1/texture cache for read-only data with spatial locality

Texture cache is optimized for 2D spatial locality (nearby (x,y) addresses hit the
same cache line). Useful for 2D matrix lookups where access patterns are not perfectly
coalesced.

```cuda
// Bind a 2D array as a texture object:
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;

cudaTextureDesc texDesc = {};
texDesc.filterMode = cudaFilterModePoint;
texDesc.readMode = cudaReadModeElementType;

cudaTextureObject_t tex;
cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

// In kernel — hardware caches nearby accesses automatically:
float val = tex2D<float>(tex, x, y);
```

*In LIO:* `rmm_input_gpu_tex` (the density matrix P_{μν}) is bound as a 2D texture
because the (i, j) access pattern in the density kernel has 2D spatial locality.
Measured `tex_cache_hit_rate` ~44% for fosfatoQMMM.

---

#### Use `__ldg()` for global loads

`__ldg()` (load through read-only data cache) routes global loads through the L1 texture
cache. Equivalent to read-only pointer restriction without requiring `const __restrict__`.
Available on SM 3.5+.

```cuda
// Standard global load (may or may not hit L1):
float val = data[i];

// Read-only cache load (always hits L1 read-only cache):
float val = __ldg(&data[i]);

// Or equivalently:
const float* __restrict__ data_ro = data;
float val = data_ro[i];  // compiler uses __ldg automatically
```

---

#### Align global memory pointers to 128-byte boundaries

The GPU memory controller fetches in 32-byte (L1 line) or 128-byte (DRAM row) chunks.
Misaligned allocations waste fetched bytes that are never used.

```cuda
// cudaMalloc guarantees 256-byte alignment — use it over posix_memalign
cudaMalloc(&ptr, size);

// For sub-allocations, manually align:
size_t aligned_size = (size + 127) & ~127;  // round up to 128 bytes
```

*In LIO:* `COALESCED_DIMENSION(d)` = `d + 32 - (d % 32)` pads matrix rows to the
next multiple of 32 elements (128 bytes for float). This ensures each row starts
at a 128-byte aligned offset for optimal coalescing.

---

#### Use `cudaMallocManaged` with `cudaMemAdvise`

Unified memory simplifies programming but can cause page faults. Use `cudaMemAdvise`
hints to pre-establish access patterns and reduce fault overhead.

```cuda
cudaMallocManaged(&data, size);

// Hint: GPU will mostly read, CPU will mostly write
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device_id);

// Hint: data is primarily used by this device
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, device_id);
```

---

### ⚡ Compute & Execution

#### Use the `__restrict__` keyword on pointers

Tells the compiler that pointer arguments do not alias (point to different memory).
Allows more aggressive load/store reordering and eliminates redundant memory reads.

```cuda
// Without restrict: compiler must reload *a each iteration (may have changed via *b)
__global__ void kernel(float* a, float* b, int N) {
    for (int i = 0; i < N; i++) a[i] = b[i] * 2.0f;
}

// With restrict: compiler can keep *b in register and vectorize the loop
__global__ void kernel(float* __restrict__ a, const float* __restrict__ b, int N) {
    for (int i = 0; i < N; i++) a[i] = b[i] * 2.0f;
}
```

---

#### Use `__builtin_expect` for branch hints

Helps the compiler generate code that minimizes branch misprediction penalties by
marking which branch is taken in the common case.

```cuda
__device__ float process(float val, bool is_boundary) {
    if (__builtin_expect(is_boundary, 0)) {  // rarely true
        return handle_boundary(val);
    }
    return val * 2.0f;  // common path — compiler keeps this in the hot path
}
```

---

#### Prefer bitwise operations over modulo/division by powers of two

Integer division and modulo are expensive (~20 cycles). For power-of-two divisors,
use bit manipulation instead.

```cuda
// Slow:
int row = index / 32;
int col = index % 32;

// Fast (equivalent when 32 is a power of two):
int row = index >> 5;   // >> 5 == / 32
int col = index & 31;   // & 31 == % 32
```

---

#### Use cooperative groups

Cooperative Groups provide flexible, safe synchronization across subsets of threads
(sub-warps, warps, blocks, or the entire grid) without `__syncthreads()` overuse.

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(float* data) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    float val = data[threadIdx.x];
    // Warp-level reduction using cooperative groups:
    for (int i = 16; i > 0; i >>= 1)
        val += warp.shfl_down(val, i);

    if (warp.thread_rank() == 0)
        atomicAdd(data, val);
}
```

---

#### Leverage Dynamic Parallelism

Allows GPU kernels to launch child kernels without returning to the CPU. Useful for
recursive algorithms (tree traversal, adaptive mesh refinement) where the work structure
is discovered at runtime.

```cuda
__global__ void parent_kernel(float* data, int level) {
    if (level > 0 && needs_refinement(data[blockIdx.x])) {
        child_kernel<<<1, 32>>>(data, level - 1);  // launch from GPU
    }
}
```

Requires `cudaDeviceSynchronize()` inside the parent or the parent must return
before accessing child results. Has significant launch overhead — not for small work.

---

#### Use `__forceinline__` for small device functions

Prevents function call overhead and register save/restore for frequently called
small helpers. Default `__device__` functions may or may not be inlined by `nvcc`.

```cuda
// Without: may generate function call overhead
__device__ float compute(float a, float b) { return a * b + 1.0f; }

// With: guaranteed inlining — call overhead eliminated
__device__ __forceinline__ float compute(float a, float b) { return a * b + 1.0f; }
```

---

### 🔄 Host-Device & Concurrency

#### Use `cudaStreamCreateWithPriority`

Assigns a scheduling priority to a stream. High-priority streams are preferentially
scheduled by the GPU when work from multiple streams is queued simultaneously.

```cuda
int low, high;
cudaDeviceGetStreamPriorityRange(&low, &high);

cudaStream_t priority_stream;
cudaStreamCreateWithPriority(&priority_stream, cudaStreamNonBlocking, high);

// Critical kernels on the priority stream:
critical_kernel<<<grid, block, 0, priority_stream>>>(args);

// Background work on a lower-priority stream:
background_kernel<<<grid, block, 0, normal_stream>>>(args);
```

---

#### Use `cudaMemsetAsync`

Asynchronous memset returns immediately, allowing the CPU to continue and overlapping
initialization with other work.

```cuda
// Synchronous — CPU blocks until memset completes:
cudaMemset(d_data, 0, N * sizeof(float));

// Asynchronous — CPU continues; memset runs in stream:
cudaMemsetAsync(d_data, 0, N * sizeof(float), stream);
// Do CPU work here while GPU initializes...
cudaStreamSynchronize(stream);  // wait only when needed
```

*In LIO:* `CudaMatrix::zero()` calls `cudaMemset` (synchronous). Switching to
`cudaMemsetAsync` on stream 0 could eliminate a synchronization point.

---

### 🛠 Tooling & Architecture

#### Design kernels to be idempotent

An idempotent kernel produces the same result when run multiple times on the same
input. This enables:
- **Easier debugging:** safe to re-run on error without state corruption
- **Restartability:** checkpoint-restart systems can re-execute without side effects
- **Testing:** compare outputs across runs to detect non-determinism

```cuda
// Non-idempotent: accumulates — running twice doubles the result
atomicAdd(&global_counter, thread_contribution);

// Idempotent: writes — running twice produces the same result
output[idx] = compute(input[idx]);
```

For accumulating kernels (like XC Fock construction), zero the output buffer in a
separate `cudaMemsetAsync` step before the accumulating kernel runs, making the
pair (memset + accumulate) idempotent as a unit.

---

## Quick Reference Table

| Technique | Tier | Dimension | Typical Speedup |
|---|---|---|---|
| Memory coalescing | 🔴 1 | Memory | 2–32× |
| Minimize H↔D transfers | 🔴 1 | Host/Dev | 2–10× |
| Warp divergence reduction | 🔴 1 | Compute | 1.5–4× |
| Block size = multiple of 32 | 🔴 1 | Compute | 1.1–2× |
| Maximize occupancy | 🔴 1 | Compute | 1.2–3× |
| float vs double | 🔴 1 | Compute | 2–32× |
| Shared memory caching | 🟡 2 | Memory | 2–10× |
| No bank conflicts | 🟡 2 | Memory | 1.5–2× |
| Pinned memory | 🟡 2 | Memory | 1.5–2× |
| Warp shuffles | 🟡 2 | Compute | 1.2–2× |
| `#pragma unroll` | 🟡 2 | Compute | 1.1–1.5× |
| Stream overlap | 🟡 2 | Host/Dev | 1.2–2× |
| CUDA Graphs | 🟡 2 | Host/Dev | 1.1–3× |
| `__ldg()` / texture cache | 🟢 3 | Memory | 1.1–1.5× |
| `__restrict__` | 🟢 3 | Compute | 1.05–1.2× |
| `__forceinline__` | 🟢 3 | Compute | 1.05–1.15× |
| `cudaMemsetAsync` | 🟢 3 | Host/Dev | 1.02–1.1× |
