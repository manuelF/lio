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

### tex2D vs `__ldg` on Pascal SM 6.1 — DO NOT replace textures

**Investigated 2026-03-20.** Replacing `tex2D` with `__ldg` in the density kernels caused
a **36% regression** in `gpu_compute_density` (1047ms → 1438ms, fosfatoQMMM).

The root cause is that `tex2D` uses the texture unit's 2D spatial locality (Morton/Z-order
tiling), which gives 82.85% L1 cache hit rate for the RMM access pattern
`data[col * stride + row]`. `__ldg` uses linear addressing on the same physical cache,
achieving only 76.48% — the 6.4 pp drop causes 9.5 pp more memory stall cycles.

**Keep tex2D for all RMM reads on Pascal.** May be revisitable on Volta+ (SM 7.0+).
See `todo/gpu/optimize_density_texture.md` for full metrics.
