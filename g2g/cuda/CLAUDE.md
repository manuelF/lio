## Profiling Procedures

### Reference test case

The standard profiling benchmark is **fosfatoQMMM** (34 QM atoms, closed-shell GGA, 25 SCF
iterations). Located at `test/LIO_test/03_fosfatoQMMM/`.

Input files: `fos.in`, `fos.xyz`, `basis`
Binary: `liosolo/liosolo`

To run it: `../../../liosolo/liosolo -i fos.in -c fos.xyz -b basis`

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
