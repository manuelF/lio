# Level Shifting: Evaluation and Results

**Status: Not viable for LIO**

Level shifting (Saunders & Hillier, 1973) was implemented and tested as an SCF
convergence optimization. The technique increases the HOMO-LUMO gap by adding a
shift to virtual orbital energies before diagonalization:

```
F'_shifted = F' + sigma * (I - C_occ * C_occ^T)
```

where `C_occ` are the previous iteration's occupied MO coefficients in the
orthonormal basis and `sigma` is the shift magnitude (Hartrees).

## Implementation

- Applied in the ON basis after DIIS extrapolation, before diagonalization
- Unshifted Fock restored after diagonalization (DIIS sees clean Fock history)
- Disabled near convergence (`good < told * 100`) to avoid slowing final iterations
- Auto-detect mode: measure HOMO-LUMO gap at iteration 3 and compute shift

## Test Results

### Manual shift sweep (fosfatoQMMM, 34 QM atoms, closed-shell GGA)

Baseline without shift: **25 iterations**, energy = -2148.6509492 A.U.

| sigma (Ha) | Iterations | Delta |
|---|---|---|
| 0.0 | 25 | baseline |
| 0.1 | 26 | +1 |
| 0.2 | 27 | +2 |
| 0.3 | 27 | +2 |
| 0.4 | 24 | -1 |
| 0.5 | 24 | -1 |

Best case saves 1 iteration (~59 ms). Non-monotonic response makes tuning fragile.

### Manual shift sweep (water, 3 atoms)

Baseline without shift: **14 iterations**.

| sigma (Ha) | Iterations | Delta |
|---|---|---|
| 0.1 | 18 | +4 |
| 0.2 | 23 | +9 |
| 0.3 | 27 | +13 |

Level shifting significantly harms easy systems.

### Auto-detect mode (`level_shift = -1`)

Measures HOMO-LUMO gap at iteration 3, applies `shift = max(0, 0.05 - gap)`.

| System | Gap at iter 3 | Auto shift | Iterations | vs baseline |
|---|---|---|---|---|
| Water | 0.383 Ha | 0.0 (none) | 14 | same |
| fosfatoQMMM | 0.007 Ha | 0.043 Ha | 26 | +1 |

Auto-detect correctly avoids shifting water but still hurts fosfatoQMMM.

### Delayed-start problem

Auto-detect must wait until iteration 3 for reliable gap measurement (iteration 1
uses the initial-guess Fock which gives unreliable gaps — water showed gap=0.009 at
iter 1 vs 0.383 at iter 3). This means the shift only applies from iteration 4
onward, disrupting DIIS history that was built without shifting. Testing at
iteration 2 was also unreliable (water gap=0.017, triggered unwanted shift).

## Why It Doesn't Work for LIO

1. **DIIS already handles convergence well.** LIO's DGELSS-based DIIS solver is
   robust to small HOMO-LUMO gaps. The fosfatoQMMM system (gap=0.007 Ha) converges
   in 25 iterations without any help from level shifting.

2. **Marginal best-case improvement.** Even the optimal manually-tuned shift only
   saves 1 iteration (~59 ms wall time) on the hardest test case. This is not worth
   the added code complexity or the risk of degrading other systems.

3. **Hurts easy systems significantly.** Any nonzero shift on well-behaved systems
   (water) drastically increases iteration count. This makes a universal default
   shift impossible.

4. **Auto-detection is unreliable.** The HOMO-LUMO gap at early iterations doesn't
   reliably predict whether level shifting will help. The delayed start (iter 3+)
   disrupts DIIS extrapolation. More sophisticated criteria (tracking oscillation
   patterns, adaptive shift scheduling) add complexity without guaranteed benefit.

5. **Non-monotonic response.** The relationship between shift magnitude and
   convergence speed is not monotonic (fosfatoQMMM: sigma=0.2 gives 27 iters,
   sigma=0.5 gives 24). There is no principled way to choose the right value.

## When Level Shifting Would Be Needed

Level shifting is primarily useful for:
- Metallic systems with zero or near-zero band gaps
- Charge-transfer complexes with orbital near-degeneracies
- Systems where SCF fails to converge entirely without it

LIO's typical workloads (organic molecules, QM/MM biomolecular systems) have
sufficient HOMO-LUMO gaps that DIIS handles convergence without difficulty.

## Recommendation

Do not pursue level shifting further. Better convergence improvements for LIO:
- **EDIIS/ADIIS** (energy-based DIIS): better early-iteration convergence than
  damping, smooth handoff to DIIS
- **Anderson/Broyden mixing**: quasi-Newton approaches that use gradient information
- **Fractional occupation (Fermi smearing)**: for genuinely metallic systems,
  smears electron occupation around the Fermi level instead of sharp cutoff
