# Test Guide

## Quick Reference

```bash
cd test

# Run everything (unit tests first, then e2e)
./run_tests.py

# Run only unit tests (builds automatically)
./run_unit.py

# Run only e2e integration tests
./run_e2e.py

# Filter by regex (works on all three scripts)
./run_tests.py --filter_rx "energy"

# List available tests without running
./run_tests.py --list

# From repo root via make
make check          # everything
make check-unit     # unit tests only
make check-e2e      # e2e tests only
```

---

## Test Tiers

### Tier 1: Unit Tests (~20 seconds)

Fast, isolated tests for individual kernels. No liosolo binary needed — each
test compiles and runs standalone against the kernel header.

**GPU kernel tests** (10 binaries, ~98 subtests):

| Test | Kernel header | What it covers |
|---|---|---|
| `accumulate_point_test` | `accumulate_point.h` | Point accumulation (closed + open shell) |
| `energy_test` | `energy.h` | Closed-shell density computation (LDA, GGA, FP precision) |
| `energy_open_test` | `energy.h`, `energy_open.h` | Open-shell density (alpha/beta) |
| `energy_derivs_test` | `energy_derivs.h` | Density derivatives (closed + open) |
| `force_test` | `force.h` | Force kernel (closed + open) |
| `functions_test` | `functions.h` | Basis function evaluation (S/P/D shells) |
| `rmm_test` | `rmm.h` | RMM update kernel |
| `rmm_gather_test` | `rmm_gather.h` | RMM gather kernel |
| `transpose_test` | `transpose.h` | Matrix transpose (float, double, vec4) |
| `weight_test` | `weight.h` | Becke partitioning weights |

**CPU conformance tests** (10 binaries, ~50 subtests):

| Test | Source header | What it covers |
|---|---|---|
| `energy_cpu_test` | `cpu_kernels.h` | CPU density vs reference implementation |
| `energy_open_cpu_test` | `kernels_reference.h` | CPU open-shell density |
| `energy_derivs_cpu_test` | `kernels_reference.h` | CPU density derivatives |
| `density_derivs_cpu_test` | `cpu_kernels.h` | CPU density derivs vs reference |
| `gga_density_cpu_test` | `cpu_kernels.h` | CPU GGA density + gradients |
| `force_cpu_test` | `kernels_reference.h` | CPU force computation |
| `functions_cpu_test` | `cpu_kernels.h` | CPU basis function evaluation |
| `rmm_cpu_test` | `cpu_kernels.h` | CPU RMM update |
| `weight_cpu_test` | `cpu_kernels.h` | CPU Becke weights |
| `partition_cpu_test` | (standalone) | CPU/GPU partition splitting logic |

### Tier 2: E2E Integration Tests (~60 seconds)

Full molecule simulations via liosolo. Require a successful `make cuda=1 cpu=1`
build. Each test runs an SCF calculation and validates physical observables
against golden `.ok` reference files.

| Test | Molecule | Checks |
|---|---|---|
| `00_agua` | Water (closed-shell) | energy, fukui, mulliken, forces, dipole |
| `01_OxyMol` | O₂ | energy, mulliken, forces |
| `02_Fe3H2O6` | Iron complex (open-shell, restart) | restart, energy |
| `03_fosfatoQMMM` | Phosphate QM/MM (34 QM atoms) | energy (1e-2), forces (1e-2), mulliken, dipole |
| `04_ECP` | ECP test | energy, dipole |
| `05_TDDFTField` | TD-DFT with field | dipole (TD), restart |
| `06_QMinPcharges` | QM in point charges | energy, forces |
| `07_TDDFTHCL` | TD-DFT on HCl | dipole (TD) |

Note: `03_fosfatoQMMM` uses wider tolerances (1e-2) because the timing-dependent
CPU/GPU rebalancer causes run-to-run FP accumulation order variation.

### Tier 3: AMBER Coupling Tests (manual)

Located in `test/AMBER_test/`. Require an AMBER installation. Not part of
automated test runs — use the `test_run*.sh` / `test_compare.sh` scripts
manually.

---

## When to Run What

### By change location

| Files changed | Minimum tests | Command |
|---|---|---|
| `g2g/cuda/kernels/energy.h` | `energy_test`, `energy_open_test` | `./run_unit.py --filter_rx "energy"` + `./run_unit.py --filter_rx "energy" --sanitize=racecheck` |
| `g2g/cuda/kernels/energy_open.h` | `energy_open_test` | `./run_unit.py --filter_rx "energy_open"` + `./run_unit.py --filter_rx "energy_open" --sanitize=racecheck` |
| `g2g/cuda/kernels/energy_derivs.h` | `energy_derivs_test` | `./run_unit.py --filter_rx "energy_derivs"` |
| `g2g/cuda/kernels/rmm.h` | `rmm_test` | `./run_unit.py --filter_rx "^rmm_test"` |
| `g2g/cuda/kernels/rmm_gather.h` | `rmm_gather_test` | `./run_unit.py --filter_rx "rmm_gather"` |
| `g2g/cuda/kernels/functions.h` | `functions_test` | `./run_unit.py --filter_rx "functions"` |
| `g2g/cuda/kernels/force.h` | `force_test` | `./run_unit.py --filter_rx "force"` |
| `g2g/cuda/kernels/weight.h` | `weight_test` | `./run_unit.py --filter_rx "weight"` |
| `g2g/cuda/kernels/transpose.h` | `transpose_test` | `./run_unit.py --filter_rx "transpose"` |
| `g2g/cuda/kernels/accumulate_point.h` | `accumulate_point_test` | `./run_unit.py --filter_rx "accumulate"` |
| `g2g/cpu/cpu_kernels.h` | all `*_cpu_test` | `./run_unit.py --filter_rx "cpu_test"` |
| `g2g/cpu/iteration.cpp` | all CPU tests + e2e | `./run_unit.py --filter_rx "cpu_test"` then `./run_e2e.py` |
| `g2g/cuda/iteration.cu` | all GPU tests + e2e | `./run_unit.py` then `./run_e2e.py` |
| `g2g/partition.cpp` | `partition_cpu_test` + e2e | `./run_unit.py --filter_rx "partition"` then `./run_e2e.py` |
| `g2g/regenerate_partition.cpp` | e2e (all) | `./run_e2e.py` |
| `g2g/init.cpp` | e2e (all) | `./run_e2e.py` |
| `g2g/classify_functions.cpp` | `functions_test` + e2e | `./run_unit.py --filter_rx "functions"` then `./run_e2e.py` |
| `g2g/global_memory_pool.*` | e2e (all) | `./run_e2e.py` |
| `lioamber/*.f90` | e2e (all) | `./run_e2e.py` |
| `lioamber/faint_cpu/*` | e2e (all) | `./run_e2e.py` |
| `lioamber/ehrensubs/*` | TD-DFT e2e | `./run_e2e.py --filter_rx "TDDFT\|Field"` |
| `lioamber/converger_subs.f90` | e2e (SCF convergence) | `./run_e2e.py` |
| `test/unit_tests/common/*` | all unit tests | `./run_unit.py` |

### Recommended workflow during development

1. **Edit a kernel header** → run its unit test immediately (~2s feedback loop):
   ```bash
   cd test && ./run_unit.py --filter_rx "energy" --no-build
   # (use --no-build if you already built; omit to auto-rebuild)
   ```

2. **Unit tests pass** → run full unit suite to check for regressions (~20s):
   ```bash
   ./run_unit.py
   ```

3. **All unit tests pass** → run e2e to verify end-to-end correctness (~60s):
   ```bash
   ./run_e2e.py
   ```

4. **Before committing** → run everything:
   ```bash
   ./run_tests.py
   ```

### Critical e2e tests for specific concerns

| Concern | Key test | What to check |
|---|---|---|
| SCF convergence changed | `03_fosfatoQMMM` | Must converge in exactly 25 iterations |
| Open-shell correctness | `02_Fe3H2O6` | Energy + restart validation |
| FP precision / DIIS | `03_fosfatoQMMM` | Energy within 1e-2 of reference |
| TD-DFT | `05_TDDFTField`, `07_TDDFTHCL` | Dipole moment trajectory |
| ECP | `04_ECP` | Energy + dipole |
| QM/MM | `03_fosfatoQMMM`, `06_QMinPcharges` | Energy + forces |

### Checking SCF convergence explicitly

The e2e check only validates final energy, not iteration count. After any
change that could affect FP behavior, also verify convergence:

```bash
cd test/LIO_test/03_fosfatoQMMM
grep "Convergence" output
# Expected: "Convergence achieved in     25 iterations."
```

---

## Script Options

All three scripts share `--filter_rx` and `--list`. Additional options:

| Script | Flag | Effect |
|---|---|---|
| `run_unit.py` | `--no-build` | Skip `make build`, run existing binaries |
| `run_unit.py` | `--sanitize` | Run GPU tests under all `compute-sanitizer` tools (memcheck, racecheck, initcheck, synccheck). Any hazard fails the test. Slow (~5× per tool) |
| `run_unit.py` | `--sanitize=<tool>` | Run one specific GPU sanitizer tool. Use `--sanitize=racecheck` in CI for the fastest reproducibility guard. |
| `run_tests.py` | `--unit-only` | Skip e2e tests |
| `run_tests.py` | `--e2e-only` | Skip unit tests |
| `run_tests.py` | `--no-build` | Passed through to unit runner |
| `run_tests.py` | `--sanitize` | Passed through to unit runner |


## Legacy

`new_tests.py` is the original e2e-only runner. It still works but its output
mixes all check results without identifying which test produced them.
`run_e2e.py` is the replacement.
