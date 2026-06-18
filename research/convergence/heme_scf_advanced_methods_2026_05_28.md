# Four candidate methods for heme SCF — math, algorithm, expectations, scope

**Status**: analysis only, none implemented. Written 2026-05-28 after the
dead-end sweep documented in
[`heme_iter_count_chaos_diagnostics_2026_05_28.md`](heme_iter_count_chaos_diagnostics_2026_05_28.md)
and [`heme_diis_stability_dead_ends_2026_05_28.md`](heme_diis_stability_dead_ends_2026_05_28.md).
Use as the reference when revisiting heme convergence performance.

## Reference baselines (current binary, `LIO_OVERLAP_INT3LU_G2G=1`)

- Heme open-shell (DZVP, charge=-1, nunp=4): **72 iters** at default OMP
  auto-tune, **72-186 iters spread** across `OPENBLAS_NUM_THREADS ∈
  {1,4,6,8}`, final E = −2829.6326 ± 1 µHa. Per-iter wall ~270 ms.
- Converger module today: ~1700 LOC across `converger_subs/`.
- Initial-guess module today: ~290 LOC in `initial_guess.f90`.

Everything below is comparison vs this baseline.

## 1. Direct minimization on orbital rotations (TRRH / QC-SCF)

### Math

Parametrize the MO coefficients as a unitary transformation of the
previous iter's MOs:

    C(X) = C_prev · exp(X)

where X is anti-Hermitian (so exp(X) is unitary) with structure
X_oo = X_vv = 0, X_ov = -X_vo^T. Only the occupied-virtual block carries
information — that's N_occ × N_virt independent real parameters
(heme: 92 × 290 ≈ 27,000 parameters).

The total energy is a smooth function E(X) on this manifold. At X = 0:

- **Gradient**: g_ai = ∂E/∂X_ai |_{X=0} = 4·F_ai^MO — the
  occupied-virtual block of the Fock matrix in MO basis. This is
  *exactly the commutator residual* [F, P] that DIIS minimizes, viewed
  differently.
- **Orbital Hessian** (also called CPHF Hessian or "B matrix"):
  H_{ai,bj} = (ε_a − ε_i) δ_ab δ_ij + 4(ai|bj) − (ab|ij) − (aj|bi)
  where (·|·) are two-electron integrals in MO basis.

**Key property**: the eigenvalues of H are bounded below by 2·HL_gap.
So even for heme (HL_gap = 0.06 Eh), H is not singular — it just has
small eigenvalues. Newton's method converges quadratically with a
well-conditioned step.

The Newton update is ΔX = −H⁻¹ g, then C_new = C_prev · exp(ΔX).
Trust-region variants (TRRH = Trust-Region Roothaan-Hall) bound
||ΔX|| ≤ τ, dynamically adjusting τ via predicted-vs-actual energy
ratio.

### Algorithm

1. Initial guess C_0.
2. At iter N:
   - Build F(P_N) in AO basis (existing infrastructure).
   - Transform to MO basis: F^MO = C_N^T F C_N.
   - Gradient: g_ai = F_ai^MO (the ov block).
   - Build Hessian H (see variants below).
   - Solve H · ΔX = −g (trust-region clipped).
   - C_{N+1} = C_N · exp(ΔX).
3. Convergence: ||g|| < ε. The gradient norm decreases monotonically
   with trust region — no oscillation, no Lyapunov amplification.

**Hessian variants** (decreasing fidelity, decreasing cost):

| Variant | Cost | Convergence |
|---------|------|-------------|
| Exact via CPHF | N⁵ build, N⁴ storage; big new integral path | Quadratic |
| Diagonal: H ≈ diag(ε_a − ε_i) | O(N²) — free (just eigenvalues) | Linear, but principled |
| L-BFGS-on-manifold | O(N²) — secant history of gradients | Superlinear in tail |

### Why this fixes heme's specific failure mode

1. **Does not require inter-iter MO alignment** — the assumption that
   killed Saunders-Hillier in MO basis. exp(X) handles arbitrary
   orbital rotations rigorously.
2. **Hessian is well-conditioned even in near-degenerate systems**:
   eigenvalues bounded below by 2·HL_gap. No DGELS-on-singular-EMAT
   chaos source.
3. **No subspace LS solve** — no Tikhonov vs basin trade-off, no
   |c|_max load-bearing concerns.
4. **Guaranteed monotonic energy decrease** with trust region — basin
   preservation is automatic.

### Expected vs current

| Metric | Current CDIIS | Direct min (diag H) | Direct min (exact H) |
|--------|---------------|---------------------|----------------------|
| Heme iters | 72-186 | ~25-40 | ~12-20 |
| OMP spread | 2.5× | <10% | <5% |
| Per-iter wall | 1.0× | 1.1-1.3× | 2-3× |
| Net wall on heme | 1.0× | ~0.4-0.6× | ~0.5-0.8× |
| Per-iter wall on easy systems | 1.0× | similar | 1.5-2× |
| Net wall on easy systems | 1.0× | similar (CDIIS already fast) | ~1.5× slower |
| Basin | preserved | preserved | preserved |

### Scope

| Component | LOC |
|-----------|-----|
| Anti-Hermitian X parametrization + exp(X) | 200 |
| MO-basis gradient builder | 100 |
| Diagonal Hessian | 50 |
| L-BFGS update (alternative to CPHF) | 250 |
| CPHF solver (alternative to L-BFGS) | 600 |
| Trust-region controller | 200 |
| Cubic line search within trust region | 100 |
| Integration into SCF loop | 300 |
| Tests (heme, Fe3H2O6, agua, fosfato) | 200 |
| **Total (diagonal Hessian path)** | **~1150 LOC** |
| **Total (exact CPHF path)** | **~1800 LOC** |

**Risks**:

- Open-shell case requires separate α/β rotations, doubling the
  parameter count.
- L-BFGS on the unitary manifold needs vector transport
  (parallel transport on tangent space) — subtle but well-established.
- Diagonal Hessian may not be enough for heme specifically (open shell
  + near-degenerate Fe-d). Pre-test on the Fe atom and Fe3H2O6 before
  committing to the full molecular path.
- exp(X) via scaling-and-squaring needs careful numerics on the
  anti-Hermitian structure to preserve unitarity to ulp.

### References

- Helgaker, Jørgensen, Olsen — *Molecular Electronic-Structure Theory*,
  Ch. 10 (orbital rotations and SCF optimization).
- Bacskay, *Chem. Phys.* 61 (1981) 385 — "A quadratically convergent
  restricted Hartree-Fock procedure" (original QC-SCF).
- Thøgersen, Olsen, *JCP* 121 (2004) 16 — trust-region augmented
  Hessian SCF.

## 2. Optimal Damping Algorithm (ODA, Cancès & Le Bris 2000)

### Math

SCF is a constrained optimization: minimize E(P) over the set

    D_N = { P : P^T = P, P^2 = P, tr(P) = N }

(idempotent density matrices with the right particle number). ODA
relaxes this to the **convex hull** of D_N (allowing fractionally
occupied states) and does line search in that convex set:

    P_{N+1} = (1 − λ*) P_N + λ* P_aufbau(F(P_N))
    λ*      = arg min_{λ ∈ [0,1]} E[ (1 − λ) P_N + λ P_aufbau ]

**Key properties**:

- E((1−λ)P_N + λP_aufbau) is **smooth** in λ — polynomial for pure HF
  (quadratic in λ since E(P) is quadratic in P and P_λ is linear in λ);
  smooth for DFT-with-XC.
- The minimum over [0, 1] is found by 1D line search (golden section
  + quadratic interpolation): ~5-10 energy evaluations per outer iter.
- **Monotonic energy decrease guaranteed** (λ = 0 is always feasible
  and gives E_{N+1} = E_N).
- **Globally convergent** to a stationary point of E on the convex
  hull, which (under mild conditions) coincides with a stationary
  point on D_N.
- Convex-combination P is generally non-idempotent during iteration —
  interpreted physically as fractionally occupied "transient" density.
  Approaches idempotent at convergence.

### Algorithm

1. Initial P_0.
2. At iter N:
   - Build F(P_N).
   - Diagonalize → aufbau density P_auf.
   - 1D line search: find λ* = arg min_{λ ∈ [0,1]} E((1−λ)P_N + λP_auf).
     Each trial λ requires evaluating E[P_λ]: build XC + Coulomb + 1e
     from P_λ. This is the cost driver.
   - P_{N+1} = (1−λ*) P_N + λ* P_auf.
3. Check convergence on ||F(P) − P · F(P)|| or ||P − P²||.

**Optimization**: cubic interpolation requires 3 energy evaluations
(at λ = 0, 0.5, 1); usually 1-2 refinement evals suffice. So ~5 energy
builds per outer iter.

### Expected vs current

| Metric | Current CDIIS | ODA |
|--------|---------------|-----|
| Heme iters | 72-186 | ~30-50 (deterministic) |
| OMP spread | 2.5× | <15% (no extrapolation, just minimization) |
| Per-iter wall | 1.0× | 2-4× (5 energy evals per iter) |
| Net wall on heme | 1.0× | ~0.8-1.5× (closer to neutral) |
| Per-iter wall on easy systems | 1.0× | 2-4× |
| Net wall on easy systems | 1.0× | ~2-4× *slower* |
| Basin | preserved | preserved (guaranteed by monotonic E) |

**Verdict**: ODA is the *robustness* lever. Slower on easy systems but
guaranteed convergent. Best as a hybrid fallback (ODA early when DIIS
oscillates, CDIIS in the tail), not a default replacement.

### Scope

| Component | LOC |
|-----------|-----|
| Line search (golden section + quadratic interp) | 250 |
| Energy evaluator for arbitrary P (uses existing Coulomb + XC + 1e) | 100 |
| Convex combination + idempotency monitor | 100 |
| Integration into SCF loop (alternative path) | 150 |
| Tests | 100 |
| **Total** | **~700 LOC** |

**Risks**:

- LIO's XC integration currently bundles E + F build together.
  Decoupling for E-only evaluations may not be cheaper than full SCF
  iter cost. May force adding an E-only-from-P path through g2g.
- Strictly speaking, ODA needs density-matrix-fractional-occupation
  handling. The convex P may have eigenvalues outside [0, 1] near
  convergence in the LIO Mulliken-style representation — may need a
  projection step.
- Late convergence is *linear* (not quadratic like CDIIS at its best).
  May plateau before hitting ||R|| < 10⁻⁶ — could need hybrid: ODA
  early, CDIIS in tail. Adds complexity.

### References

- Cancès & Le Bris, *IJQC* 79 (2000) 82 — "Can we outperform the DIIS
  approach for electronic structure calculations?"
- Kudin, Scuseria, Cancès, *JCP* 116 (2002) 8255 — EDIIS, with ODA
  semantics and the convex-combination construction.

## 3. Modified Broyden / Anderson mixing in density space

### Math

Treat SCF as fixed-point iteration

    P_new = F(P_old) = aufbau(eig(F(P_old)))

— an implicit nonlinear function. Want P* with F(P*) = P*. Define
residual R(P) = F(P) − P. Solve R(P) = 0 via quasi-Newton.

Build a low-rank approximation of the inverse Jacobian
J⁻¹ = (dR/dP)⁻¹ from secant pairs:

    Δs_n = P_n − P_{n-1},   Δy_n = R_n − R_{n-1}

**Broyden's first update** (rank-one):

    J⁻¹_{n+1} = J⁻¹_n + ((Δs_n − J⁻¹_n Δy_n) ⊗ Δy_n) / (Δy_n^T Δy_n)

**Anderson mixing** (multi-secant Broyden equivalent): solve LS over
recent residuals,

    min_{c_i}  || Σ_i c_i R_{n-i} ||²    s.t.   Σ c_i = 1

then update P_{n+1} = Σ c_i P_{n-i} + β Σ c_i R_{n-i}.

**This is structurally similar to DIIS** but acts on (P, R) pairs
instead of (F, [F,P]).

**Eyert's modified Broyden** (used heavily in plane-wave DFT — VASP,
Quantum Espresso): adds Tikhonov regularization to the LS solve based
on ||Δy|| weighting; restarts when condition number blows up.

### Algorithm

1. Initial P_0; evaluate R_0 = F(P_0) − P_0.
2. At iter N:
   - Save Δs_N = P_N − P_{N-1}, Δy_N = R_N − R_{N-1}.
   - Build approximate J⁻¹ from last k secant pairs (typical k = 5-15).
   - Step: ΔP = −α · J⁻¹ · R_N, where α ∈ (0, 1] is a damping factor
     (line search optional).
   - P_{N+1} = P_N + ΔP.
   - Evaluate R_{N+1}.
3. Convergence: ||R|| < ε.

**Restart heuristic**: if the secant pair magnitude
||Δs|| / ||Δy|| becomes pathological (condition number > 10⁸), drop
history and restart with steepest descent.

### Where it might break the same way as windowed DIIS

This is the critical concern. Anderson mixing IS essentially DIIS in
primal space, with a LS solve over the same kind of subspace. If
heme's "load-bearing" wild extrapolation (|c| = 1.87) is needed for
convergence, Anderson is likely to want the same kind of large
coefficients — which means the same sensitivity to ulp shifts.

The advantage over CDIIS: Anderson works with **gradient differences**
(Δy) rather than absolute residuals. This is incrementally more stable
when the residual is small (near convergence) because cancellation in
Δy preserves accuracy. But the wild-phase coefficients are similarly
unconstrained.

**Verdict from prior research**: the |c|_max cap, Tikhonov, windowed
DIIS, and DGELSD have all failed on heme. Modified Broyden lives in
the same family — high likelihood of same failure mode.

### Expected vs current

| Metric | Current CDIIS | Modified Broyden |
|--------|---------------|------------------|
| Heme iters | 72-186 | 60-150 (best case) — or fails like windowed DIIS |
| OMP spread | 2.5× | unknown — could be similar |
| Per-iter wall | 1.0× | 1.0× (no extra integral cost) |
| Risk of failure on heme | (baseline) | **HIGH** — same class as rejected approaches |

### Scope

| Component | LOC |
|-----------|-----|
| Broyden inverse-Jacobian update | 250 |
| History buffer (parallel to fockm) | 100 |
| Damping factor / line search | 100 |
| Restart logic | 50 |
| Integration / SCF dispatch | 100 |
| Tests | 100 |
| **Total** | **~700 LOC** |

**Honest assessment**: least promising of the four. Structurally close
to approaches already known to fail on heme. Worth knowing about for
completeness, but should NOT be pursued first.

### References

- Eyert, *J. Comput. Phys.* 124 (1996) 271 — modified Broyden for DFT.
- Walker & Ni, *SIAM J. Numer. Anal.* 49 (2011) 1715 — "Anderson
  acceleration for fixed-point iterations."
- Banerjee, Suryanarayana, Pask, *Chem. Phys. Lett.* 647 (2016) 31 —
  periodic Pulay (Anderson with restart).

## 4. SAD-d initial guess (Superposition of Atomic Densities, d-shell)

### Math

The SCF starting density is currently built from the 1-electron
Hamiltonian guess (LIO `initial_guess = 0`): diagonalize H^core
(kinetic + nuclear attraction, no e-e), occupy lowest N eigenvalues.

For molecules with transition metals, this is a terrible guess because
H^core doesn't know about the d-shell structure — the lowest
eigenvalues of H^core on Fe are roughly hydrogenic-like, not
d-occupied.

**SAD** constructs the initial guess as a sum of pre-computed atomic
densities:

    P_init = Σ_A π_A · P_atom_A · π_A^T

where P_atom_A is the spherically averaged ground-state density of
atom A in its local basis, and π_A is the projection from atom A's
basis to the molecular basis (built from overlap matrices).

For **SAD-d** specifically (transition metals): the atomic density
respects the d-shell occupation pattern:

- Fe (Z = 26): 3d⁶ 4s² ground state, spin-restricted average →
  P_atom has fractional occupations on each of the 5 d-orbitals
  (averaged over m_l) summing to 6 electrons.
- Sum over all atoms preserves total electron count.

### Why it should help heme

The wild iters 1-12 are the d-shell reorganization phase — the SCF
is figuring out which d-orbitals to occupy. With SAD-d, **this phase
is largely skipped** because the initial density is already
approximately correct on each atom.

HL_gap data from
[`heme_iter_count_chaos_diagnostics_2026_05_28.md`](heme_iter_count_chaos_diagnostics_2026_05_28.md)
confirms this is where the chaos originates:

| iter | HL_gap (Eh) | comment |
|------|-------------|---------|
| 1 | 0.020 | catastrophically narrow |
| 7 | 0.035 | still near-degenerate |
| 12 | 0.068 | basin locked in (above converged) |
| 72 | 0.062 | converged |

With SAD-d, iter 1 HL_gap should already be ~0.05 Eh (close to
converged) — no wild phase, no need for |c| = 1.87 extrapolations,
no Lyapunov amplification of ulp shifts.

The existing LIO **aufbau guess** (`initial_guess = 1`) is conceptually
similar but cruder — it distributes electrons by orbital type (s/p/d)
without doing actual atomic SCF. From [`MEMORY.md`](../../../) note on
fosfato (closed shell): "aufbau: 26-27 iters vs baseline 25 — slower
despite better starting energy." On easy systems, the 1e guess + DIIS
is already close enough that a better initial density doesn't beat
the tuning. **For transition metals, the gap should be much larger** —
that's exactly the case where 1e guess fails badly.

### Algorithm

**One-time per element** (precompute or runtime-cached):

1. Build atom's 1-electron Hamiltonian + minimal-basis SCF.
2. Run spherically averaged Hartree-Fock (or DFT) for the atom.
3. Average over rotations / m_l (so density is spherically symmetric
   in the atom's frame).
4. Store P_atom(Z) keyed by element + basis.

**At molecular SCF start**:

1. For each atom A:
   - Identify its basis-function indices in the molecular basis.
   - Place P_atom(Z_A) at those indices (block on diagonal of P_init).
2. Normalize: P_init ← P_init · (N_electrons / tr(P_init)).
3. Optional: project to nearest idempotent matrix (sharpen the guess).
4. Call SCF with P_init as the starting density (existing
   infrastructure).

**Open-shell extension**: spherically averaged atomic density doesn't
distinguish α / β. For SAD-d on heme (NUNP = 4, S = 2), distribute the
unpaired electrons over the d-shell of Fe according to Hund's rule
(highest spin in lowest orbital energy).

### Expected vs current

| Metric | Current 1e guess + CDIIS | SAD-d + CDIIS |
|--------|--------------------------|---------------|
| Heme iters | 72-186 | ~25-45 (wild phase avoided) |
| OMP spread | 2.5× | ~1.3-1.5× (smaller perturbation window) |
| Per-iter wall | 1.0× | 1.0× |
| Init time | ~50 ms | ~70 ms (one-shot precompute or table lookup) |
| Net wall on heme | 1.0× | ~0.35-0.6× |
| Behavior on easy systems (agua, fosfato) | baseline | same or better |
| Basin preservation | yes | yes |

### Scope

| Component | LOC |
|-----------|-----|
| Atomic minimal-basis SCF driver (could reuse molecular SCF) | 200 |
| Spherical averaging routine | 80 |
| Open-shell Hund's-rule d-shell distributor | 100 |
| Atomic basis tables (per-element configs) | 100 (data) |
| Atomic → molecular basis projection | 100 |
| Integration into `initial_guess.f90` (new option 2) | 80 |
| Tests (TM systems: Fe3H2O6, heme, Zn timers) | 150 |
| **Total** | **~810 LOC** (mostly mechanical) |

**Risks**:

- Atomic Fe SCF can itself be tricky (open shell, near-degenerate).
  Need a robust atomic SCF — could use the same direct minimization
  approach proposed in §1 for atoms, but for SAD-d alone a simple
  damping-only SCF on the spherically averaged atom should converge
  in ~10 iters.
- For unusual oxidation states (e.g., Fe(IV) in some heme
  intermediates), the neutral-atom SAD-d guess may be the wrong charge
  state. Mitigation: SAD-d gives global electron count = sum of neutral
  atoms; then post-scale to match total charge as a starting point.
- LIO already has `initial_guess = 1` (aufbau) which is similar in
  spirit. Need to verify on heme that SAD-d materially beats aufbau;
  if not, the engineering cost isn't justified.

### References

- Lehtola, *JCTC* 15 (2019) 1593 — "Assessment of initial guesses for
  self-consistent field calculations: superposition of atomic
  potentials."
- Almlöf, Faegri, Korsell, *JCC* 3 (1982) 385 — original SAD.
- Van Lenthe et al., *JCP* 125 (2006) 244111 — orbital occupation
  conventions for SAD with open-shell atoms.

## Comparison table

| Method | Net wall on heme | Wall on easy | OMP spread | Basin safe | Scope LOC | Risk |
|--------|-------------------|--------------|------------|------------|-----------|------|
| Current CDIIS+rollback | 1.0× | 1.0× | 2.5× | ✓ | — | (baseline) |
| **SAD-d guess** | **~0.4-0.6×** | similar | ~1.3× | ✓ | ~800 | low — addresses root cause (wild phase) |
| **Direct min (diag H)** | ~0.4-0.6× | ~1.2× | <10% | ✓ | ~1150 | medium — needs validation on heme specifically |
| Direct min (exact H) | ~0.5-0.7× | ~1.5-2× | <5% | ✓ | ~1800 | high — major CPHF integral pipeline rewrite |
| ODA | ~0.8-1.5× | ~2-4× | <15% | ✓ | ~700 | medium — slower on easy systems |
| Modified Broyden | unknown | similar | unknown | unknown | ~700 | **high — same family as rejected approaches** |

## Recommendation

### If picking one: SAD-d initial guess

Highest expected-value-per-LOC.

1. **Addresses the actual root cause.** Heme chaos is amplified ulp
   shifts over many iterations. SAD-d cuts the iteration count roughly
   in half by skipping the wild d-shell reorganization phase, which
   cuts the ulp-amplification window proportionally.
2. **Doesn't touch the SCF accelerator** — CDIIS keeps doing what it
   does. No risk of regressing easy systems.
3. **No mathematical landmines** — it's a starting condition, not an
   algorithm change. Even with implementation bugs, worst case it's no
   worse than current 1e guess (existing fallback path).
4. **Useful even if direct minimization is implemented later** —
   direct min also benefits from a good initial guess. Investments
   compose.

### If picking two: SAD-d + direct minimization (diagonal Hessian)

Together they should give ~20-30 iters on heme with <10% OMP spread,
at ~1900 LOC investment over both. The direct minimization piece is
the more speculative bet (depends on whether diagonal Hessian is
enough for heme — pre-test on Fe atom and Fe3H2O6 first), but with
SAD-d as fallback, even if direct min struggles, the speedup from the
guess alone is captured.

### If picking zero today

Ship the current diagnostic infrastructure
(`verbose > 3` `[scf_diag]` log), accept iter chaos, benchmark on
basin energy across OMP. Treat heme iter count as a noisy timer for
now, and reconsider once a different optimization pressure (e.g., a
much bigger system where 70-200 iter spread becomes prohibitive in
wall time) makes the engineering investment worth it.

## Cross-references

- [`heme_iter_count_chaos_diagnostics_2026_05_28.md`](heme_iter_count_chaos_diagnostics_2026_05_28.md)
  — the dead-end sweep that produced the constraints used here.
- [`heme_diis_stability_dead_ends_2026_05_28.md`](heme_diis_stability_dead_ends_2026_05_28.md)
  — algebraic DIIS safeguards rejected previously.
- [`heme_dgels_rank_deficiency_2026_05_28.md`](../../) (in
  `~/.claude/.../memory/` — the DGELS rank-deficiency root cause).
