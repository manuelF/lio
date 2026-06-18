# Fortran Codebase (lioamber)

Research on modernizing and optimizing the Fortran 90 layer that drives
the SCF loop, handles I/O, and calls into the C++/CUDA engine.

**Status (2026-06-18).** The Fortran-side per-iteration hot paths are
**exhausted**: the SCF loop is BLAS/LAPACK-floored at M=364, and ~95% of the
fosfato/TD wall lives in separate compilation units (`libg2g.so` and
`libopenblas`), not in lioamber's own FFLAGS — so structural compiler flags
cannot move the needle (see `lto_blas_interposition_dead_end`). The remaining
levers are algorithmic (iter-count, see `../convergence/`) or live in `g2g`.

Landed Fortran-side wins (details on the `optimizations` branch +
`MEMORY.md`): int3mem OpenMP parallelization, Cholesky for the G matrix,
converger direct P'_ON DGEMMs, allocation hoisting, the int3lu↔g2g overlap
(`LIO_OVERLAP_INT3LU_G2G`, closed/open-shell) with thread auto-tuning, and the
int3lu GPU-resident cuBLAS offload (gate lifted — see
[`../cpu/int3lu_gpu_offload_2026_06_10.md`](../cpu/int3lu_gpu_offload_2026_06_10.md)).

## Files

| File | Status | Summary |
|------|--------|---------|
| [makefile_optimizations_and_interface_lint_2026_06_16.md](makefile_optimizations_and_interface_lint_2026_06_16.md) | DONE | Makefile flag audit (10+ opts) + interface-lint program. implicit-interface **809→0** + -Waliasing 0; both **promoted to default FFLAGS** as guards; only -Warray-temporaries (305) stays `lint=1`-gated. All e2e PASS. Profiling: 9.2ms/iter g2g idle is the algorithmic pole. |
| [lto_blas_interposition_dead_end_2026_06_18.md](lto_blas_interposition_dead_end_2026_06_18.md) | REJECTED | `-flto` / `-fexternal-blas` / `-fno-semantic-interposition` measured end-to-end: no wall-clock win, two perturb FP-sensitive heme/fosfato. Root cause: ~95% of fosfato/TD wall is in `libg2g.so` + `libopenblas`, not lioamber FFLAGS. Closes the structural-flag branch. |
| [overlap_autotune_2026_05_03.md](overlap_autotune_2026_05_03.md) | DONE | Auto-tunes OMP/BLAS threads from physical core count for the int3lu↔g2g overlap; `g2g/hardware_topo.{h,cpp}` module; no manual env vars needed. |
| [int3lu_bandwidth_bound_multizn_2026_06_01.md](int3lu_bandwidth_bound_multizn_2026_06_01.md) | DONE (diag) / threading REJECTED | int3lu (0.54s/call on multiZn) is DRAM-bandwidth-bound BLAS-2 streaming 5.8GB cool/cools 2×/call; CPU threading measured-dead (flat 1→8 threads); overlap can't hide it. Real lever = GPU-resident cuBLAS offload, since shipped (see `../cpu/int3lu_gpu_offload`). |
| [../convergence/heme_scf_hotpath_and_sad_d_refutation_2026_05_29.md](../convergence/heme_scf_hotpath_and_sad_d_refutation_2026_05_29.md) | DONE / DEAD ENDS | Heme SCF Fortran/BLAS per-iter hot paths **exhausted**: Fock integrals GPU-bound under overlap; base change/DIIS already BLAS3; thread count absorbed by spare cores; diagonalization **basis-locked** (dsyevr-full 72→119 iters). **SAD-d refuted** (no-op under vcinp=t; aufbau d-shell guess 391 iters vs restart 72). |
