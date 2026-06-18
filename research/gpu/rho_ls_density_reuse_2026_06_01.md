# Rho linear search — XC energy via endpoint-density reuse

**Status:** OPEN (designed + profiled, not yet implemented)
**Impact:** HIGH — targets ~90% of GPU time on the LinearSearchRho case
**Case:** `test/LIO_test/11_LinearSearchRho` (fe3h2o, DZVP, open shell, charge 3,
nunp 1, vcinp restart, `Rho_LS=2`, `told=1e-8`). Run with
`LIO_OVERLAP_INT3LU_G2G=1`.

## Profile (baseline, 53 iters, ~8.4 s wall)

Timer summary (`timers=2`):

| Bucket | Time | % of Iteration |
|--------|------|----------------|
| Iteration | 7.89 s | 100% |
| **Rho update & check** | **6.95 s** | **88%** |
| → Rho linear search | 6.95 s | — |
| →→ **LS - XC g2g** | **6.17 s** | **89% of LS** |
| →→ LS - int3lu | 0.76 s | 11% of LS |
| Fock integrals (int3lu, main) | 0.51 s | 6.4% |
| SCF - Fock Diagonalization | 0.26 s | 3.3% |

The linear search dominates the whole run. It does **726 energy evaluations**
(66 cycles × 11 λ), each calling `give_me_energy` →
`int3lu(energy_only)` + `g2g_solve_groups(COMPUTE_ENERGY_ONLY)`.

### Why so many evals
`converger_ls.f90:rho_linear_calc` evaluates `E(λ)` for λ = 0…1 in 11 steps,
and re-runs the whole 11-point scan up to 5 cycles per SCF step (adapting
`Pstepsize`). Near convergence the energy is **flat within float32 XC noise**
(~1e-5 Eh on a −1720 Eh total), so `line_search`'s `minloc` lands on a random
index and burns cycles chasing noise (step 51 did 6 cycles = 66 evals).

### nsys kernel breakdown of the energy-only XC call
```
gpu_compute_density_opened<float,0,0>   81.1%   (big groups)
gpu_compute_density_opened<float,0,1>    9.1%   (small groups)
gpu_gather_rdm_open                       4.0%
gpu_accumulate_point_open  (XC funct.)    3.7% + 0.3%
```
**~94% of GPU time is computing the density (+grad/hess) at grid points from
the density matrix; only ~4% is the actual XC functional.** `compute_rmm` is
already `false` for `COMPUTE_ENERGY_ONLY`, so the Fock build is NOT the cost.

## The lever: endpoint-density reuse

The line search evaluates the XC energy of `P(λ) = (1−λ)·P0 + λ·P1`, where
`P0 = rho_lambda0` and `P1 = rho_lambda1` are **fixed for the entire SCF step**
(set once in `rho_linear_calc` before the cycle loop; the cycle loop only reads
them). The grid-point density is *linear* in P:

```
ρ(r, λ)   = (1−λ)·ρ0(r)   + λ·ρ1(r)
∇ρ(r, λ)  = (1−λ)·∇ρ0(r)  + λ·∇ρ1(r)   (same for the Hessian terms)
```

So compute `ρ0, ∇ρ0, ρ1, ∇ρ1` **once per SCF step** (2 density passes over all
groups) and reuse them for *every* λ and *every* cycle — each λ eval then needs
only a cheap pointwise blend + the 4% XC functional. For step 51 that turns
66 full density contractions into 2 + 66 cheap functional passes.

Expected payoff: density contraction is 94% of the energy-only call, so the LS
XC time drops by roughly the reuse ratio. Pessimistically (functional 30%) ~2.7×;
at the measured ~94% density, ~5× on cycle-heavy steps. Plausible whole-run wall
≈ 8.4 s → ~4–5 s.

## Implementation sketch
- **C++/CUDA (`iteration.cu`, `partition.*`)**: two new GPU paths —
  `solve_ls_setup` (gather + `gpu_compute_density_opened` into *two* sets of
  per-group cached buffers: `partial_densities_a/b`, `dxyz_a/b`, `dd1_a/b`,
  `dd2_a/b` for endpoint 0 and endpoint 1) and `solve_ls_energy(λ)` (a cheap
  blend kernel `(1−λ)·end0 + λ·end1` into the working buffers, then the
  *unmodified* `gpu_accumulate_point_open` energy kernel). Keeping the XC
  functional kernel untouched preserves its validated FP behaviour.
- **init.cpp**: upload both endpoint density matrices (alpha+beta) and expose
  `g2g_solve_ls_setup_` / `g2g_solve_ls_energy_` `extern "C"` shims.
- **Fortran (`converger_ls.f90`)**: call setup once after `rho_lambda0/1` are
  fixed, then replace the per-λ `g2g_solve_groups` inside `give_me_energy` with
  the cheap `g2g_solve_ls_energy`. (int3lu stays; or fit its exactly-quadratic
  `E2(λ)` from 3 points — secondary, only 11%.)

## Validation protocol (MANDATORY — chaotic trajectory)
This case is Lyapunov-divergent in iter count like heme: thread sweep gives
**46/49/53/54 iters and final energy −1610.6068…−1610.6077 across OMP={1,4,8}**.
So **do not judge by a single run**. Accept the change iff:
- median iters across OMP={1,4,8} stays ≈ 50 (not systematically higher), and
- final energy stays in the basin −1610.607 ± 5e-4 Eh.

Blending two float32 densities is the same perturbation class as the
thread-count shuffle already tolerated, so it should stay in-basin — but it
MUST be measured, not assumed.

## Phase 1 — Fortran scaffold (DONE, trajectory-neutral)

Refactored `rho_linear_calc` (`converger_ls.f90`): the endpoint fixing is now
wrapped in a `LS - endpoint setup` timer (where the 2 GPU density passes will
go), and every λ evaluation routes through a single internal seam
`ls_lambda_energy(λ)` (blend endpoints → `give_me_energy`). Added an
`[LS-evals]` per-step counter.

- **Bit-neutral**: OMP=1 reproduces the baseline *exactly* (53 iters,
  −1610.6071028). OMP=4/8 differ run-to-run but stay in-basin — verified
  inherent: a repeat OMP=4 run reproduced the baseline's 46 / −1610.6068013
  exactly. So the refactor changed no math; multi-thread variance is the
  documented chaos.
- **Eval-count opportunity quantified**: 53 steps → **812 full energy evals**
  (per-step: 10×2, 15×13, 9×24, 6×35, 1×46, 1×57, 1×68). With endpoint reuse
  these collapse to **~106 density contractions (2/step) + 812 cheap functional
  passes** → ~5.6× on the `LS - XC g2g` bucket (6.0 s), wall ≈ 8.4 → ~3.3 s.
- **Blast radius**: only `11_LinearSearchRho` reaches `rho_linear_calc`
  (`do_rho_ls` is gated on `rho_LS > 1`; Heme is `Rho_LS=1`). No closed-shell
  e2e covers it — the closed branch is algebraically identical + bit-neutral
  but ships e2e-untested.

## Phase 2 — GPU numerics spike (NEXT, make-or-break, do BEFORE plumbing)

The scaffold de-risked the call structure; it tested **nothing** about the one
real assumption: that in float32, `dens(P(λ)) ≈ (1−λ)·dens(P0) + λ·dens(P1)`
holds tightly enough to stay in-basin (direct-contract vs blend-then-reuse
round a few ulp apart). Write ~50 LOC throwaway: for one mid-trajectory step,
compute Ex via the reuse path (density for P0,P1 into 2 buffers, blend at a few
λ, run *unmodified* `accumulate_point`) and compare per-λ Ex to current
`g2g_solve_groups(blended P)`. Agreement ~1e-5 relative ⇒ build the real path.
Disagreement ⇒ learned for 50 LOC, and the fix (FULL_DOUBLE endpoint storage)
would kill the speedup — so gate on this first.

When building the real path: `give_me_energy` needs a λ argument (int3lu still
needs the blended density *vector* for E2; only the XC call takes λ and hits the
reuse path). The `[LS-evals]` print should be gated on verbose or dropped before
landing on master.

## Phase 2 — GPU numerics spike (DONE, PASS)

Built the real GPU primitives (`iteration.cu`): `gpu_ls_blend_s/v4` kernels +
`PointGroupGPU::ls_compute_endpoint(slot)` (density from global RDM into
endpoint buffers), `ls_accumulate_working`, `ls_blend_accumulate_energy(λ)`
(blend endpoints → working → `accumulate_point`), and `ls_direct_energy`
(validation). Driver `Partition::ls_spike` + `g2g_ls_spike_` (in init.cpp — note
the TWO `partition` globals: `::partition` in init.cpp is the populated one,
`G2G::partition` in partition.cpp is empty), gated by `LIO_LS_SPIKE=1`.

Result on a mid-trajectory step (109 GPU groups, `cpu_threads=0` so reuse covers
all groups), GPU-group XC energy, reuse vs direct (blend-then-contract):

| λ | rel diff |
|---|---|
| 0.00 / 1.00 | 0 (pure endpoint) |
| 0.25 / 0.50 / 0.75 | **1.2e-8 / 9.9e-9 / 1.2e-8** (abs ~1.3e-6 Eh) |

**PASS by 3 orders** vs the ~1e-5 bar; the perturbation is *below* the ~3e-6
float32 XC noise floor and ~1e-3 basin tolerance the SCF already tolerates
run-to-run. fp32 `dens(P(λ)) ≈ (1−λ)dens(P0)+λdens(P1)` is sound. FULL_DOUBLE
endpoint storage NOT needed. **Cleared to build the production path.**

Note: this case is `cpu_threads=0` (all groups GPU). For hybrid partitions
(`cpu_threads>0`) the CPU groups need handling too — production should either
add CPU-group endpoint reuse or fall back to the direct path unless the
partition is GPU-only.

## Phase 3 — production wiring + the tolerance wall (DONE)

Wired end-to-end: `g2g_ls_setup_`/`g2g_ls_energy_` (init.cpp) →
`Partition::ls_setup`/`ls_energy` (returns active=1 only when *every* group is
on GPU, i.e. `cpu_threads==0`, else fall back to `g2g_solve_groups`). Fortran
`give_me_energy` gained `(dlambda, use_reuse)`; `converger_ls.f90` calls
`g2g_ls_setup` once per step and routes each λ eval through `g2g_ls_energy` when
active. Fused blend kernel `gpu_ls_blend_fused` (1 launch/group/λ).

**Timing (isolated, 13 evals incl. setup): 2.12× per-eval** (direct 97.8ms vs
reuse 46.1ms). Launch overhead on the per-group blend+accumulate caps it below
the 5.6× density-elimination projection.

**The wall: systematic bias breaks tight-`told` convergence.** End-to-end at the
benchmark's `told=1e-8`: **no convergence in 1000 iters** (basin-correct but
ΔRho oscillates at ~1e-7, never reaching 1e-8). Root cause: blend-then-contract
vs contract-then-blend differ by a *systematic, same-sign* ~1.3e-6 Eh on Ex
(spike: reuse consistently less-negative at intermediate λ — a float32
contraction-order artifact in the GGA |∇ρ| terms). A non-zero-mean bias injects
a constant Blambda kick every step → a ΔRho floor ~1e-7. (Random thread noise is
zero-mean → averages out → still reaches 0; that's why baseline tolerates 1e-3
thread jitter but reuse's 1.3e-6 *systematic* error doesn't.) The per-eval
numerics spike (1e-8 rel) structurally could not see this trajectory
accumulation.

**But it's a real win at normal tolerance.** At the *documented default*
`told=1e-6` (`fe3h2o_t6.in`): reuse **68 iters / 4.72s, in basin** vs direct
51 iters / 7.92s → **1.68× wall** (+33% iters from the perturbation, more than
paid back by ~2× cheaper iters).

**Shipped gated (correct everywhere):** reuse activates only for open shell +
all-GPU partition + `told >= 1e-6` (≈7× above the measured ~1.5e-7 floor); the
`told=1e-8` benchmark auto-falls-back to direct (baseline 53 iters bit-exact
preserved). Env escape `LIO_LS_NOREUSE=1`. Net: the *task case* (told=1e-8)
cannot safely benefit (its tolerance is below the float32 contraction floor),
but general normal-tolerance open-shell Rho_LS runs get ~1.68×.

## Dead end recorded
**Energy-floor gate (skip LS when `Enew−Elast < 1e-5`) — REJECTED.** Tried
widening the `else` (Blambda=1.0) branch to absorb sub-noise energy increases.
Result: **53 → 82 iters, 8.4 s → 11.4 s.** The flat-energy late steps carry
load-bearing rho motion that the damping needs; skipping them reshuffles the
chaotic trajectory the wrong way. Any *trajectory-changing* shortcut is unsafe
here. The density-reuse win is safe because it preserves the algorithm (same
λ scan, same accept logic) and only makes each eval cheaper.
