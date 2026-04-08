## Profiling Procedures

### Hardware and tooling constraints

- **ncu (Nsight Compute) requires SM 7.0+** — cannot be used on this hardware
- **nvprof** is the primary profiling tool for this project

### Reference test case

The standard profiling benchmark is **fosfatoQMMM** (34 QM atoms, closed-shell GGA, 25 SCF
iterations). Located at `test/LIO_test/03_fosfatoQMMM/`.

Input files: `fos.in`, `fos.xyz`, `basis`
Binary: `liosolo/liosolo`

### Project-specific gotchas

- Use the normal release build for profiling (not `dbg=1` — debug disables optimizations).
- Always measure wall time **without** nvprof first (nvprof adds ~10-20% overhead).
- `liosolo` must be called directly, NOT via `run.sh`. nvprof does not profile child
  processes by default, so `./run.sh` produces an empty profile.
- **Save profiles once, query many times** (`nvprof -o file.nvvp` then `nvprof -i file.nvvp`).
  Avoids re-running the full simulation for each metric query.
- `--metrics` queries require re-running with instrumentation (~5-10× slower). They CANNOT
  be extracted from saved `.nvvp` files. Only collect for specific kernels you're investigating.

### Time budget framework

Parse profiles into these categories:

| Category | How to measure |
|---|---|
| GPU kernel time | `--print-gpu-summary`, sum all kernel times |
| GPU memcpy time | `--print-gpu-summary`, sum `[CUDA memcpy *]` rows |
| cudaMalloc+cudaFree | `--print-api-summary`, sum those two rows |
| cudaStreamSync | `--print-api-summary` |
| CPU overhead | wall_time minus the above |

Separate SCF kernels (called per iteration × 25) from post-SCF kernels
(`gpu_qmmm_forces`, `gpu_coulomb_forces`, `gpu_qmmm_fock` — called once).

---

## Architecture-Specific Notes

### tex2D vs `__ldg` — DO NOT replace textures (dead end)

Tested on **Pascal SM 6.1** (2026-03-20) and **Ampere SM 8.6** (2026-04-07).
Both show regressions for different architectural reasons:

- **Pascal**: 36% regression. `__ldg` loses 6.4 pp L1 hit rate vs tex2D's 2D Morton tiling.
- **Ampere**: 2.5× regression. L1 hit rate *improved* (+9.5 pp) but software address
  computation (`row * stride + col`) in the tight inner loop collapses compute throughput
  from 94% to 60%. The ~14 FP ops per fetch cannot absorb 2 extra integer ops per fetch.

**Exception**: `gpu_compute_density_derivs` improved 55% with `__ldg` on Ampere (cooperative
shared memory loads decouple address computation from compute), but at 5.6% of GPU time
it's not worth a mixed approach.

**Keep tex2D for all RMM reads.** See `../../research/gpu/optimize_density_texture.md`
for full ncu metrics on both architectures.
