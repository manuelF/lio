# Initial Guess Evaluation (2026-04-17)

**Status:** TESTED — aufbau is slower on fosfato; default 1-e guess is optimal
**Impact:** Negative (26-27 iter vs 25)

## Background

LIO has two initial guess modes in `lioamber/initial_guess.f90`:
- `initial_guess=0` (default): diagonalize 1-e Hamiltonian → orbitals with no
  electron-electron repulsion, then form density from those orbitals.
- `initial_guess=1`: atomic aufbau — superpose atomic densities per Iz/shell
  (similar in spirit to SAD / Superposition of Atomic Densities).

Hypothesis (coming in): aufbau should reduce iteration count because its
starting energy is much closer to the converged SCF energy. Typical textbook
result.

## Measurement

fosfatoQMMM, RTX 3080 Ti + Ryzen 7 5800X3D, timers=2, 3 warm runs each:

| Mode | Step-1 energy | Iter count | Total time |
|------|---------------|------------|------------|
| `initial_guess=0` (1-e) | -1440.65 | 25 | 1.77-1.89s |
| `initial_guess=1` (aufbau) | -1969.75 | 26-27 | 1.93-2.08s |
| Converged energy | -2148.65 | — | — |

Aufbau starts ~530 au closer to truth but takes 1-2 **more** iterations.

## Why aufbau loses

The 1-e diagonalization gives **correct orbital shapes** from the non-interacting
Fock operator: the MOs from the bare nuclear attraction + kinetic are already
in the right symmetry classes and spatial regions. Only the electron-electron
repulsion is missing — DIIS heals that by iterating the mean field.

Aufbau superposes atomic densities. Charges are right but orbital topology is
wrong: there are no molecular MOs, just overlapping atomic shells. DIIS then
has to heal *both* the missing repulsion and the wrong orbital shapes. Despite
the lower initial energy, the density-matrix trajectory is further from the
converged fixed point.

LIO's damping+DIIS(ndiis=30, GOLD=10) appears specifically tuned for the 1-e
guess trajectory. The float32 DIIS noise floor (~2e-6 in rho_diff, only 3×
below `told=1e-6`) means any convergence machinery change has a narrow
stability window.

## Implications for iteration reduction

Iteration reduction is **not a practical lever** on fosfato:

| Approach | Estimated save | Feasibility |
|----------|---------------|-------------|
| Aufbau guess | **-2 iters (worse)** | TESTED dead end |
| Earlier DIIS activation (lower good_cut) | 1-2 iters | risky with float32 noise |
| Density-based DIIS / ADIIS / EDIIS | 2-4 iters | multi-week implementation, uncertain with float32 |
| Second-order SCF (Newton-Raphson) | 5-10 iters | months of work, requires orbital Hessian |

LIO's 25-iter convergence on fosfato is near the practical floor given the
float32 GPU kernels driving DIIS. Further iteration reduction requires
either (a) moving to double-precision GPU kernels (breaks dual-row kernel
design, massive regression) or (b) rewriting the converger with a less
noise-sensitive algorithm.

## When to revisit

- Systems with slow convergence (open-shell, metals, near-degenerate HOMO-LUMO)
  may show different relative behavior; re-test aufbau on `02_Fe3H2O6`.
- If float32 is ever replaced by true mixed-precision (Tensor Cores on
  newer GPUs), DIIS sensitivity relaxes and aufbau may start to win.
- For MD runs with small geometry changes, density-extrapolation from
  previous SCFs would dominate either guess — this is a separate lever.
