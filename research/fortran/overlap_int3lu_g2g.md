# Overlap int3lu with g2g_solve_groups (XC Fock)

**Status:** OPEN (top priority as of 2026-04-17)
**Expected impact:** 150-270ms save (6-15% wall) on fosfatoQMMM
**Estimated difficulty:** Medium-High

## Problem

Per SCF iteration, `SCF.f90:485-498` runs two independent Fock contributions
strictly sequentially:

```
call int3lu(E2, Pmat_vec, Fmat_vec2, Fmat_vec, Gmat_vec, Ginv_vec, &
            Hmat_vec, open, MEMO)            ! Coulomb fit + Fock (CPU-only BLAS)
call g2g_solve_groups(0, Ex, 0)              ! XC Fock (GPU + CPU partition)
```

On fosfatoQMMM/RTX 3080 Ti (M=804 in packed form):
- `int3lu`: ~10.7ms/iter, multi-threaded OpenBLAS (dgemv/sgemv/dspmv)
- `g2g_solve_groups`: ~20ms/iter, GPU kernel + CPU partition threads

During `int3lu` the GPU is idle and the partition worker threads sit in an
OpenMP barrier. During `g2g_solve_groups` the OpenBLAS workers that served
`int3lu` are idle. This accounts for a meaningful slice of the 40% OpenMP
barrier time seen in the perf flat profile.

Critical observation: **the two routines are independent**. They read the same
`Pmat_vec` (density) but do not read each other's Fock output. The final Fock
is just `Fmat = Hmat + Coulomb + XC`, and both contributions are commutative
additions.

## Strategy

Run `int3lu` and `g2g_solve_groups` concurrently in two OpenMP sections.

### Buffer split

int3lu does `Fmat(1:MM) = Hmat(1:MM)` then accumulates Coulomb (subm_int3lu.f90:149).
If g2g_solve_groups writes XC additively into `Fmat_vec`, we need separate
buffers so writes don't race:

```fortran
! Use a separate XC Fock buffer; merge after the overlap barrier.
double precision, allocatable, save :: Fmat_xc(:)
if (.not. allocated(Fmat_xc)) allocate(Fmat_xc(MM))
Fmat_xc = 0.0d0

!$omp parallel sections default(shared) num_threads(2)
!$omp section
    call openblas_set_num_threads(n_blas)   ! e.g. 4
    call int3lu(E2, Pmat_vec, Fmat_vec2, Fmat_vec, Gmat_vec, Ginv_vec, &
                Hmat_vec, open, MEMO)
    call openblas_set_num_threads(1)
!$omp section
    call g2g_solve_groups_into(0, Ex, 0, Fmat_xc)   ! new entry point
!$omp end parallel sections

Fmat_vec(1:MM) = Fmat_vec(1:MM) + Fmat_xc(1:MM)
```

The merge is an O(MM) DAXPY equivalent — negligible (MM ~323k doubles).

### Thread management (the tricky part)

Right now OpenBLAS and libgomp share the same libgomp runtime (Ubuntu
`openblas-openmp`). Without care, nested parallelism either over-subscribes
(15 threads × 15 threads) or disables itself entirely.

The split we want during overlap:
- Section 1 (int3lu): 1 main thread + `n_blas` OpenBLAS worker threads (e.g. 4)
- Section 2 (g2g_solve): remaining 10-11 threads for the CPU partition team

Implementation:
1. Query total threads = `omp_get_max_threads()` at entry.
2. Use `omp_set_nested(.true.)` + `omp_set_max_active_levels(2)`.
3. Inside section 1: `openblas_set_num_threads(n_blas)`; restore to 1 after.
4. Inside section 2: the existing `g2g_solve_groups` `#pragma omp parallel for`
   over groups picks up the remaining threads.

Note: the OPENBLAS_NUM_THREADS sweep from 2026-04-17 was **outside** any
parallel region — all calls had the full thread pool. Inside a
`parallel sections`, behavior is different and this becomes a real lever.

### Alternative: GPU offload of int3lu Coulomb fit

Instead of CPU/GPU overlap, move the Ginv·af solve onto the GPU via cuBLAS.
int3lu's two dominant BLAS calls are:
- `DGEMV('T', M=Md, N=something, Ginv, ...)`: projecting density → fit coeffs
- Second DSPMV for Fock assembly

A GPU path would require moving `Ginv_vec` to device memory (once, persistent)
and streaming Pmat_vec each iter. Likely 2-3× faster per BLAS call, but the
overlap approach is simpler and captures most of the win without changing
int3lu's interface.

## Risks / Validation

1. **Energy drift from ordering of FP additions.** The merge
   `Fmat += Fmat_xc` happens once per iter, so the bit pattern differs from
   the original (where XC was added *inside* g2g to a Coulomb-filled Fmat).
   With float32 DIIS noise at 2e-6, this ordering change is within the noise
   floor; verify with full e2e.

2. **`g2g_solve_groups` may not have an "into buffer" entry point.** Look at
   `g2g/init.cpp`, `g2g/partition.cpp` for the existing Fock write path and
   the FortranMatrix bridging. Cheapest option: snapshot Fmat_vec before,
   diff after — but that serializes. Better: add a scratch buffer parameter
   or use a thread-local Fock accumulator on the C++ side.

3. **OpenBLAS/libgomp interaction under nested sections.** Test both
   `openblas-openmp` (shared libgomp) and `openblas-pthread` (independent
   pool). Measure total thread count inside each section with
   `omp_get_num_threads()`.

4. **`MEMO` path in int3lu reads `cool/cools/kkind/kkinds` from int3mem.**
   These are read-only during int3lu — safe to share between threads.

## Validation plan

1. Implement behind an env var toggle `LIO_OVERLAP_INT3LU_G2G=1` so old path
   remains bit-exact default.
2. Run fosfato warm 5× with both paths; compare energy (should match within
   7th decimal), convergence iter count, and Coulomb fit + Fock + XC Fock
   total time.
3. Run full e2e suite on the overlap path.
4. If stable: make it the default and remove the toggle.

## Call sites (so an implementer can find the code quickly)

- `lioamber/SCF.f90:487` — int3lu call
- `lioamber/SCF.f90:497` — g2g_solve_groups(0, Ex, 0) call (XC Fock)
- `lioamber/faint_cpu/subm_int3lu.f90:72-74` — int3lu signature
- `g2g/partition.cpp` — `g2g_solve_groups` implementation (C++ side)
- `g2g/init.cpp` — Fortran bridge for Fmat/RMM pointers

## Estimation breakdown

Per iter, the current critical path on the Coulomb+XC portion is:
```
int3lu (10.7ms) + g2g_solve_groups (20ms) = 30.7ms/iter
```

Under perfect overlap:
```
max(int3lu, g2g_solve_groups) = 20ms/iter
```

Savings per iter: 10.7ms. Over 25 iters: **267ms**. That's the ceiling.

Realistic expectation (with merge overhead, imperfect thread split, first-iter
warmup): **150-250ms = 8-14% wall reduction** on fosfato; larger saves on
systems where int3lu is closer to or exceeds g2g time.
