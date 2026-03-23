# SCF Convergence and Numerical Precision

Research on the SCF convergence machinery, DIIS solver, and the interaction
between float32 GPU kernels and convergence behavior. This is critical context
for anyone modifying GPU kernels — float32 noise is only 3x below the convergence
threshold, so many "improvements" actually break convergence.

**Read [scf_data_flow.md](scf_data_flow.md) first** for the full picture of how
data flows through the SCF loop.

## Files

| File | Status | Summary |
|------|--------|---------|
| [scf_data_flow.md](scf_data_flow.md) | REF | Complete SCF data flow: GPU/CPU pipeline, DIIS, float32 noise analysis |
| [converger_optimizations.md](converger_optimizations.md) | DONE | DGELS → DGELSS fix for rank deficiency; ndiis=30 necessity explained |
| [scf_optimizations.md](scf_optimizations.md) | REF | SCF loop architecture; Fock construction cost breakdown |
| [level_shifting_evaluation.md](level_shifting_evaluation.md) | REJECTED | Level shifting tested and ruled out for LIO (minimal benefit) |

## Key Constraints (read before touching convergence-sensitive code)

1. **Float32 noise floor is ~2e-6 in rho_diff** — only 3x below `told=1e-6`
2. **ndiis=30 is required** (not the typical 6-12) because float32 noise needs large DIIS history
3. **GOLD=10 is optimal** — reducing damping amplifies noise, worsens convergence
4. **Kahan summation is harmful** — changes float32 bit patterns, disrupts DIIS trajectory
5. **`__launch_bounds__` changes FP results** — different register allocation reorders FMA instructions
6. **Warp shuffle reductions are safe** IF summation order matches the original volatile pattern
