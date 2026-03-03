# Optimization: Replace Manual Loops with BLAS (CPU)

## Summary
The CPU implementation of `solve_closed` and `solve_opened` in `g2g/cpu/iteration.cpp`
contains two phases with BLAS-replaceable operations:

1. **Density phase**: `cpu_compute_density_lda` and `cpu_compute_density_gga` compute
   $\rho(p) = \sum_{ij} R_{ij} F_i(p) F_j(p)$. Effective GEMM: $Y = F \cdot R$,
   then $\rho_p = \text{dot}(F_p, Y_p)$.

2. **RMM update phase**: `cpu_update_rmm` computes
   $\delta R_{ij} = \sum_p w_p \cdot F_i(p) \cdot F_j(p)$. Effective SYRK:
   $R \mathrel{+}= (F_w)^T F_w$ where $F_w(p,i) = F(p,i)\sqrt{w_p}$.

## Current State (post-refactoring)

As of commit `84fa3a4b`, the inner loops have been extracted into free functions in
`g2g/cpu/cpu_kernels.h`:
- `cpu_compute_density_lda(fv, rmm, m)` → scalar density per point
- `cpu_compute_density_gga(fv, gx, ..., rmm, m)` → `GGADensity<T>` struct
- `cpu_compute_density_derivs(...)` → force derivs per atom
- `cpu_update_rmm(fvr, fvc, factors, npoints)` → scalar RMM element

These functions are O(M) or O(M²) scalar loops — correct reference implementations.
The BLAS replacement lands in `iteration.cpp` **at the call sites** (replacing the
outer loops that invoke these functions), not inside them.

The original claim of "triple nested loops" is slightly inaccurate: the refactored
code calls `cpu_update_rmm` once per (bi, bj) pair (O(M²) calls × O(P) each = O(M²P))
and `cpu_compute_density_lda` once per point (O(P) calls × O(M²) each = O(M²P)).
Both have the same asymptotic complexity; BLAS replaces the outer dispatch structure.

## Proposal

### Phase 1 — RMM Update with BLAS SSYRK (Highest ROI)

Replace the double loop over (bi, bj) pairs calling `cpu_update_rmm` with a single
SSYRK call:
```cpp
// Step 1: weight function values F_w(p,i) = function_values(i,p) * sqrt(factors[p])
HostMatrix<scalar_type> F_weighted(npoints, group_m);
for (int p = 0; p < npoints; ++p)
  for (int i = 0; i < group_m; ++i)
    F_weighted(p, i) = function_values_transposed(i, p) * sqrtf(factors_rmm[p]);
    // Note: function_values_transposed is group_m × npoints, row i = function i

// Step 2: R += F_w^T * F_w (lower triangle SYRK, N=group_m, K=npoints)
// HostMatrix is row-major (asArray() gives row-major data)
cblas_ssyrk(CblasRowMajor, CblasLower, CblasTrans,
            group_m,   // N: output matrix dimension
            npoints,   // K: contraction dimension
            1.0f,      // alpha
            F_weighted.asArray(), group_m,  // A (npoints×group_m row-major), LDA
            1.0f,      // beta
            rmm_output_local.asArray(), group_m);  // C (group_m×group_m), LDC
```

Note on layout: `HostMatrix(i,j) = data[j*width+i]` is column-major in (i,j). When
passing as `CblasRowMajor` array, verify the stride (LDA) matches actual storage.
The safest approach: use `CblasColMajor` with the natural HostMatrix layout.

**Expected speedup: 5–20× for the RMM phase** (BLAS SSYRK uses AVX2 vectorized
dot products with full register blocking vs scalar loops).

### Phase 2 — Density with BLAS SSYMM + Dot Products

Replace the per-point loop calling `cpu_compute_density_lda` with a single SSYMM:
```cpp
HostMatrix<scalar_type> Y(npoints, group_m);  // P×M intermediate
cblas_ssymm(CblasRowMajor, CblasRight, CblasLower,
            npoints, group_m,
            1.0f, rmm_input.asArray(), group_m,     // Symmetric R (M×M)
            function_values.asArray(), group_m,      // F (npoints×group_m, P×M)
            0.0f, Y.asArray(), group_m);             // Y = F·R

// Density: rho[p] = dot(F_p, Y_p)
for (int p = 0; p < npoints; ++p)
  density_out[p] = cblas_sdot(group_m, function_values.row(p), 1, Y.row(p), 1);
```

For GGA, Y is reused for gradient contractions — same pattern as GPU GEMM proposal.

### Makefile Integration

BLAS is already available when `intel=2` (MKL). For OpenBLAS add:
```makefile
ifeq ($(cpu),1)
  CXXFLAGS += -DCPU_BLAS
  LDFLAGS  += $(if $(filter 2,$(intel)),-lmkl_rt,-lopenblas)
endif
```

Wrap with compile-time guard in `iteration.cpp`:
```cpp
#ifdef CPU_BLAS
  #include <cblas.h>
#endif
```

Use a runtime threshold for small groups:
```cpp
if (group_m > 16 && npoints > 32) { /* BLAS path */ }
else { /* scalar path (cpu_kernels.h functions) */ }
```

## Impact

For CPU-mode simulations (`cuda=0 cpu=1`):
- RMM SSYRK: **5–20× speedup** for RMM phase (M≥20, P≥64).
- Density SSYMM: **3–10× speedup** for density phase (M≥20).
- End-to-end CPU mode: **3–8× overall** for large systems.
- Small groups (M<16): BLAS overhead negates benefit — scalar path stays.

The original "10x–50x" estimate is overly optimistic: it applies to large BLAS-friendly
matrix sizes (M=500+). For typical LIO group sizes (M=10–60), expect **5–20× per phase**.

## Difficulty Assessment
**Medium** (accurate)

- Floating-point ordering changes: energy differences < 1e-6 Hartree expected.
  Run `agua` and `fosfato` tests; they compare against `.ok` files with numerical
  tolerances (not bit-exact).
- Layout mapping: `HostMatrix(i,j) = data[j*width+i]` is column-major. Map to BLAS
  `CblasColMajor` for simplicity, or use `CblasRowMajor` with transposed arguments.
  The LDA parameter must match the actual data stride.
- `function_values_transposed` (used by RMM loop) is M×P. Its `row(i)` returns
  a pointer to P consecutive values for function i — matches `CblasColMajor` A
  with LDA=npoints.

Files to modify:
- `g2g/cpu/iteration.cpp`: Replace RMM and density outer loops with BLAS calls.
- `g2g/Makefile` (root) and `g2g/Makefile.cuda`: Add BLAS library flags for `cpu=1`.

## Estimations
- RMM phase speedup: **5–20×** vs scalar loops (BLAS SSYRK with AVX2).
- Density phase speedup: **3–10×** (BLAS SSYMM).
- End-to-end CPU mode: **3–8×** for large systems (M≥30, P≥128).
- Small systems (M<10): **1–1.5×** (overhead dominated).
- Implementation: 3–5 days including layout verification and test validation.
