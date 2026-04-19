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
| [initial_guess_evaluation.md](initial_guess_evaluation.md) | REJECTED | Aufbau guess is slower than 1-e default on fosfato (26-27 vs 25 iters); iteration reduction not a lever |
| [scf_loop_profile_2026_03_28.md](scf_loop_profile_2026_03_28.md) | DONE | Full nsys+perf profile: 49% CPU idle, 840ms sync bottleneck, allocation hoisting |
| [numerical_stability_investigation_2026_04_17.md](numerical_stability_investigation_2026_04_17.md) | DONE | Variance characterization; FULL_DOUBLE build broken (6.6 mHa offset); initcheck clean |
| [reproducibility_investigation_2026_04_19.md](reproducibility_investigation_2026_04_19.md) | DONE | Float32 noise sources quantified (3e-7 floor + 4× rebalance amp); FULL_DOUBLE brokenness scoped (agua fine, fosfato broken — size-dependent, memcheck clean) |

## Key Constraints (read before touching convergence-sensitive code)

1. **Float32 noise floor is ~2e-6 in rho_diff** — only 3x below `told=1e-6`
2. **ndiis=30 is required** (not the typical 6-12) because float32 noise needs large DIIS history
3. **GOLD=10 is optimal** — reducing damping amplifies noise, worsens convergence
4. **Kahan summation is harmful** — changes float32 bit patterns, disrupts DIIS trajectory
5. **`__launch_bounds__` changes FP results** — different register allocation reorders FMA instructions
6. **Warp shuffle reductions are safe** IF summation order matches the original volatile pattern
7. **`full_double=1` build is broken at scale** (2026-04-19 update): agua (3 atoms) is correct and bit-exact, fosfato (34 atoms) converges to wrong energy (6.6 mHa off) or fails to converge at all. Memcheck clean. Threshold between 3 and 34 atoms not yet bisected.
