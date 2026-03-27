#!/usr/bin/env python3
"""
Cost model benchmark for LIO CPU/GPU partition optimizer.

Fits and compares old (P×M²) vs new (block-aware) cost models for CPU standalone,
GPU standalone, and hybrid parallel execution. Recommends constants for the C++ code.

Usage:
    cd test/LIO_test/03_fosfatoQMMM
    python3 ../../../g2g/bench/costmodel_tune.py \
        -i fos.in -c fos.xyz -b basis

    # Re-analyze saved data (no liosolo run):
    python3 ../../../g2g/bench/costmodel_tune.py \
        -i fos.in -c fos.xyz -b basis \
        --skip-measure --cpu-data /tmp/allcpu.txt --gpu-data /tmp/allgpu.txt

Requires: LIO built with `make cuda=1 cpu=1`, liohome.sh sourced.
"""

import argparse
import math
import os
import sys

# Reuse utilities from splitpoint_tune.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splitpoint_tune import (
    find_liosolo, make_verbose_input, run_liosolo,
    parse_perfmodel, parse_balance
)

DENSITY_BLOCK_SIZE = 64  # from g2g/common.h


def block_height(M):
    """GPU grid height: ceil(M / (2 * DENSITY_BLOCK_SIZE))."""
    return (M + 2 * DENSITY_BLOCK_SIZE - 1) // (2 * DENSITY_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Fitting utilities
# ---------------------------------------------------------------------------

def fit_ols(X, y):
    """Multivariate OLS: y = X @ beta.  X is list of rows, y is list of floats.
    Returns (beta, r2).  No intercept — add a column of 1s if you want one."""
    n = len(y)
    k = len(X[0])
    # X^T X
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    # X^T y
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    # Solve via Gaussian elimination (good enough for k<=4)
    beta = _solve(XtX, Xty)
    if beta is None:
        return [0.0] * k, 0.0
    y_pred = [sum(X[i][a] * beta[a] for a in range(k)) for i in range(n)]
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    mean_y = sum(y) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return beta, r2


def _solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for col in range(n):
        # Pivot
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < 1e-30:
            return None
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]
    # Back-substitute
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def compute_residuals(X, y, beta):
    """Return list of (predicted, actual, residual, pct_error)."""
    res = []
    for i in range(len(y)):
        pred = sum(X[i][a] * beta[a] for a in range(len(beta)))
        actual = y[i]
        residual = actual - pred
        pct = abs(residual) / actual * 100 if actual > 1 else 0
        res.append((pred, actual, residual, pct))
    return res


# ---------------------------------------------------------------------------
# Hybrid parallel simulation (reused from splitpoint_tune concept)
# ---------------------------------------------------------------------------

def simulate_hybrid(groups, cpu_model_fn, gpu_model_fn, split_cost_fn, n_cpu):
    """For each possible split, compute max(cpu_bottleneck, gpu_total).

    split_cost_fn(P, M) returns a sortable cost for determining split order.
    Groups with cost <= threshold go to CPU, others to GPU.

    Returns list of (threshold, wall, cpu_bottleneck, gpu_total).
    """
    import heapq

    # Compute costs and sort by split_cost ascending
    indexed = []
    for P, M, _, t_cpu, t_gpu in groups:
        sc = split_cost_fn(P, M)
        cpu_pred = cpu_model_fn(P, M)
        gpu_pred = gpu_model_fn(P, M)
        indexed.append((sc, cpu_pred, gpu_pred, t_cpu, t_gpu))
    indexed.sort(key=lambda x: x[0])

    # Start with everything on GPU
    gpu_total_pred = sum(gp for _, _, gp, _, _ in indexed)
    gpu_total_actual = sum(tg for _, _, _, _, tg in indexed)

    results = []

    # All-GPU case (threshold = 0)
    results.append((0, gpu_total_pred, 0.0, gpu_total_pred,
                     gpu_total_actual, 0.0, gpu_total_actual))

    cpu_preds = []
    cpu_actuals = []

    for i, (sc, cp, gp, tc, tg) in enumerate(indexed):
        # Move group i from GPU to CPU
        gpu_total_pred -= gp
        gpu_total_actual -= tg
        cpu_preds.append(cp)
        cpu_actuals.append(tc)

        # LPT bin-pack CPU groups
        sorted_cpu = sorted(cpu_preds, reverse=True)
        sorted_cpu_actual = sorted(cpu_actuals, reverse=True)

        def lpt_max(times, n_threads):
            if not times or n_threads < 1:
                return 0.0
            bins = [0.0] * n_threads
            for t in sorted(times, reverse=True):
                bins[min(range(n_threads), key=lambda x: bins[x])] += t
            return max(bins)

        cpu_bn_pred = lpt_max(cpu_preds, n_cpu)
        cpu_bn_actual = lpt_max(cpu_actuals, n_cpu)

        wall_pred = max(cpu_bn_pred, gpu_total_pred)
        wall_actual = max(cpu_bn_actual, gpu_total_actual)

        results.append((sc, wall_pred, cpu_bn_pred, gpu_total_pred,
                         wall_actual, cpu_bn_actual, gpu_total_actual))

    return results


def _make_verbose_local(src, dst):
    """Copy input file to dst in CWD, ensuring verbose=4."""
    import re
    with open(src) as f:
        content = f.read()
    if re.search(r'verbose\s*=', content, re.IGNORECASE):
        content = re.sub(r'verbose\s*=\s*\d+', 'verbose = 4', content,
                         flags=re.IGNORECASE)
    else:
        content = re.sub(r'(&end)', r' verbose = 4\n\1', content,
                         flags=re.IGNORECASE)
    with open(dst, 'w') as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LIO cost model benchmark: old (P×M²) vs block-aware")
    parser.add_argument("--input", "-i", required=True, help="LIO input file")
    parser.add_argument("--coords", "-c", required=True, help="Coordinate file")
    parser.add_argument("--basis", "-b", required=True, help="Basis set file")
    parser.add_argument("--binary", default=None, help="Path to liosolo binary")
    parser.add_argument("--skip-measure", action="store_true",
                        help="Read from --cpu-data/--gpu-data instead of running")
    parser.add_argument("--cpu-data", type=str, default=None)
    parser.add_argument("--gpu-data", type=str, default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="OMP_NUM_THREADS (default: auto)")
    parser.add_argument("--sweep", action="store_true",
                        help="Validate hybrid prediction with wall-time sweep")
    args = parser.parse_args()

    binary = args.binary or find_liosolo()
    if not binary:
        print("ERROR: Cannot find liosolo. Set --binary or source liohome.sh.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Using liosolo: {binary}")

    # =====================================================================
    # Phase 1: Data Collection
    # =====================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Data Collection")
    print("=" * 70)

    cpu_data = []
    gpu_data = []

    if args.skip_measure:
        if args.cpu_data:
            with open(args.cpu_data) as f:
                cpu_data = parse_perfmodel(f.read())
            print(f"  Loaded {len(cpu_data)} CPU groups from {args.cpu_data}")
        if args.gpu_data:
            with open(args.gpu_data) as f:
                gpu_data = parse_perfmodel(f.read())
            print(f"  Loaded {len(gpu_data)} GPU groups from {args.gpu_data}")
    else:
        # Create verbose copy in CWD with short name (liosolo has 20-char arg limit).
        # Use a distinct name to avoid overwriting the original.
        verbose_input = "_bench_v4.in"
        _make_verbose_local(args.input, verbose_input)
        omp = str(args.threads or os.cpu_count() or 4)

        print("  Running all-CPU (LIO_SPLIT_COST=very large)...")
        cpu_out, cpu_wall = run_liosolo(
            binary, verbose_input, args.coords, args.basis,
            split_cost=999999999999999, env_extra={"OMP_NUM_THREADS": omp})
        cpu_data = parse_perfmodel(cpu_out)
        print(f"  Collected {len(cpu_data)} groups, wall={cpu_wall:.1f}s")

        print("  Running all-GPU (LIO_SPLIT_COST=0)...")
        gpu_out, gpu_wall = run_liosolo(
            binary, verbose_input, args.coords, args.basis,
            split_cost=0)
        gpu_data = parse_perfmodel(gpu_out)
        print(f"  Collected {len(gpu_data)} groups, wall={gpu_wall:.1f}s")

        # Clean up temp input file
        if os.path.exists(verbose_input):
            os.remove(verbose_input)

    if not cpu_data or not gpu_data:
        print("ERROR: No perfmodel data. Build with cuda=1 cpu=1 and verbose=4.",
              file=sys.stderr)
        sys.exit(1)

    n = min(len(cpu_data), len(gpu_data))
    if len(cpu_data) != len(gpu_data):
        print(f"  WARNING: count mismatch CPU={len(cpu_data)} GPU={len(gpu_data)}, using {n}")

    # Merge into unified list: (P, M, cost, t_cpu, t_gpu)
    groups = []
    for i in range(n):
        P, M, cost, t_cpu = cpu_data[i]
        _, _, _, t_gpu = gpu_data[i]
        groups.append((P, M, cost, t_cpu, t_gpu))

    m_values = sorted(set(M for _, M, _, _, _ in groups))
    p_values = sorted(set(P for P, _, _, _, _ in groups))
    print(f"\n  {n} groups: P={min(p_values)}..{max(p_values)}, "
          f"M={min(m_values)}..{max(m_values)}")

    # =====================================================================
    # Phase 2: CPU Standalone Model
    # =====================================================================
    print("\n" + "=" * 70)
    print("Phase 2: CPU Standalone Model")
    print("=" * 70)

    # Filter tiny groups (timer noise)
    cpu_filtered = [(P, M, c, tc, tg) for P, M, c, tc, tg in groups if tc > 5.0]
    nc = len(cpu_filtered)

    # Old model: T_cpu = α × P×M²
    X_cpu_old = [[P * M * M] for P, M, _, _, _ in cpu_filtered]
    y_cpu = [tc for _, _, _, tc, _ in cpu_filtered]
    beta_cpu_old, r2_cpu_old = fit_ols(X_cpu_old, y_cpu)

    print(f"\n  Old model: T_cpu = α × P×M²")
    print(f"    α     = {beta_cpu_old[0]:.6e} µs/(P×M²)")
    print(f"    R²    = {r2_cpu_old:.4f}  ({nc} groups)")

    # New model: T_cpu = α × P×M + β × P×M²
    X_cpu_new = [[P * M, P * M * M] for P, M, _, _, _ in cpu_filtered]
    beta_cpu_new, r2_cpu_new = fit_ols(X_cpu_new, y_cpu)

    print(f"\n  New model: T_cpu = α₁×P×M + α₂×P×M²")
    print(f"    α₁ (P×M)  = {beta_cpu_new[0]:.6e}")
    print(f"    α₂ (P×M²) = {beta_cpu_new[1]:.6e}")
    print(f"    R²         = {r2_cpu_new:.4f}")
    print(f"    Δ R²       = {r2_cpu_new - r2_cpu_old:+.4f}")

    # Worst residuals (old model)
    res_cpu_old = compute_residuals(X_cpu_old, y_cpu, beta_cpu_old)
    res_cpu_old_sorted = sorted(enumerate(res_cpu_old), key=lambda x: -x[1][3])
    print(f"\n  Worst 5 residuals (old CPU model):")
    print(f"  {'Grp':>5} {'P':>6} {'M':>4} {'Predicted':>10} {'Actual':>10} {'Error%':>8}")
    for idx, (pred, actual, resid, pct) in res_cpu_old_sorted[:5]:
        P, M = cpu_filtered[idx][0], cpu_filtered[idx][1]
        print(f"  {idx:>5} {P:>6} {M:>4} {pred:>10.1f} {actual:>10.1f} {pct:>7.1f}%")

    # =====================================================================
    # Phase 3: GPU Standalone Model
    # =====================================================================
    print("\n" + "=" * 70)
    print("Phase 3: GPU Standalone Model")
    print("=" * 70)

    # Skip first group (cold start)
    gpu_filtered = [(P, M, c, tc, tg) for P, M, c, tc, tg in groups[1:] if tg > 5.0]
    ng = len(gpu_filtered)

    # Old model: T_gpu = γ + α × P×M²
    X_gpu_old = [[1.0, P * M * M] for P, M, _, _, _ in gpu_filtered]
    y_gpu = [tg for _, _, _, _, tg in gpu_filtered]
    beta_gpu_old, r2_gpu_old = fit_ols(X_gpu_old, y_gpu)

    print(f"\n  Old model: T_gpu = γ + α × P×M²")
    print(f"    γ (overhead) = {beta_gpu_old[0]:.1f} µs")
    print(f"    α (slope)    = {beta_gpu_old[1]:.6e} µs/(P×M²)")
    print(f"    R²           = {r2_gpu_old:.4f}  ({ng} groups)")

    # New model: T_gpu = γ + β₁×M² + β₂×P×ceil(M/128)×M
    X_gpu_new = [[1.0, M * M, P * block_height(M) * M]
                 for P, M, _, _, _ in gpu_filtered]
    beta_gpu_new, r2_gpu_new = fit_ols(X_gpu_new, y_gpu)

    print(f"\n  New model: T_gpu = γ + β₁×M² + β₂×P×ceil(M/128)×M")
    print(f"    γ  (fixed overhead)  = {beta_gpu_new[0]:.1f} µs")
    print(f"    β₁ (RMM setup, M²)  = {beta_gpu_new[1]:.6e}")
    print(f"    β₂ (compute, P×bh×M)= {beta_gpu_new[2]:.6e}")
    print(f"    R²                   = {r2_gpu_new:.4f}")
    print(f"    Δ R²                 = {r2_gpu_new - r2_gpu_old:+.4f}")

    # Staircase analysis: groups near M=128 boundary
    near_128 = [(P, M, c, tc, tg) for P, M, c, tc, tg in gpu_filtered
                if 120 <= M <= 136]
    if near_128:
        print(f"\n  Staircase analysis near M=128 ({len(near_128)} groups):")
        print(f"  {'P':>6} {'M':>4} {'bh':>3} {'Actual':>9} {'Old pred':>10} "
              f"{'Old err%':>9} {'New pred':>10} {'New err%':>9}")
        for P, M, _, _, tg in sorted(near_128, key=lambda x: x[1]):
            bh = block_height(M)
            old_pred = beta_gpu_old[0] + beta_gpu_old[1] * P * M * M
            new_pred = (beta_gpu_new[0] + beta_gpu_new[1] * M * M +
                        beta_gpu_new[2] * P * bh * M)
            old_err = abs(tg - old_pred) / tg * 100 if tg > 0 else 0
            new_err = abs(tg - new_pred) / tg * 100 if tg > 0 else 0
            print(f"  {P:>6} {M:>4} {bh:>3} {tg:>9.1f} {old_pred:>10.1f} "
                  f"{old_err:>8.1f}% {new_pred:>10.1f} {new_err:>8.1f}%")

    # Worst residuals (old vs new)
    res_gpu_old = compute_residuals(X_gpu_old, y_gpu, beta_gpu_old)
    res_gpu_new = compute_residuals(X_gpu_new, y_gpu, beta_gpu_new)
    res_old_sorted = sorted(enumerate(res_gpu_old), key=lambda x: -x[1][3])
    print(f"\n  Worst 5 residuals (old GPU model):")
    print(f"  {'Grp':>5} {'P':>6} {'M':>4} {'bh':>3} {'Old pred':>10} "
          f"{'Actual':>10} {'Old%':>7} {'New%':>7}")
    for idx, (pred, actual, resid, pct) in res_old_sorted[:5]:
        P, M = gpu_filtered[idx][0], gpu_filtered[idx][1]
        new_pct = res_gpu_new[idx][3]
        bh = block_height(M)
        new_pred = (beta_gpu_new[0] + beta_gpu_new[1] * M * M +
                    beta_gpu_new[2] * P * bh * M)
        print(f"  {idx:>5} {P:>6} {M:>4} {bh:>3} {pred:>10.1f} "
              f"{actual:>10.1f} {pct:>6.1f}% {new_pct:>6.1f}%")

    # =====================================================================
    # Phase 4: Hybrid Parallel Model
    # =====================================================================
    print("\n" + "=" * 70)
    print("Phase 4: Hybrid Parallel Model")
    print("=" * 70)

    omp_threads = args.threads or int(os.environ.get(
        "OMP_NUM_THREADS", os.cpu_count() or 4))
    n_cpu = max(1, omp_threads - 1)
    print(f"\n  Simulating: {n_cpu} CPU threads + 1 GPU thread")

    # Define model functions for hybrid simulation
    def cpu_model_old(P, M):
        return beta_cpu_old[0] * P * M * M

    def gpu_model_old(P, M):
        return max(0, beta_gpu_old[0] + beta_gpu_old[1] * P * M * M)

    def cpu_model_new(P, M):
        return max(0, beta_cpu_new[0] * P * M + beta_cpu_new[1] * P * M * M)

    def gpu_model_new(P, M):
        bh = block_height(M)
        return max(0, beta_gpu_new[0] + beta_gpu_new[1] * M * M +
                   beta_gpu_new[2] * P * bh * M)

    def split_by_pm2(P, M):
        return P * M * M

    # Run hybrid simulation with old model
    hybrid_old = simulate_hybrid(groups, cpu_model_old, gpu_model_old,
                                 split_by_pm2, n_cpu)
    # Find best predicted split (old model)
    best_old = min(hybrid_old, key=lambda x: x[1])

    # Run hybrid simulation with new model
    hybrid_new = simulate_hybrid(groups, cpu_model_new, gpu_model_new,
                                 split_by_pm2, n_cpu)
    best_new = min(hybrid_new, key=lambda x: x[1])

    # Find best actual split
    best_actual = min(hybrid_old, key=lambda x: x[4])

    print(f"\n  {'Model':<12} {'Best thresh':>12} {'Pred wall':>10} "
          f"{'Actual wall':>12} {'Pred bottleneck':>16}")

    def bottleneck(cpu, gpu):
        return "CPU" if cpu > gpu else "GPU"

    print(f"  {'Old':<12} {best_old[0]:>12.0f} {best_old[1]:>10.0f} "
          f"{best_old[4]:>12.0f} {bottleneck(best_old[2], best_old[3]):>16}")
    print(f"  {'New':<12} {best_new[0]:>12.0f} {best_new[1]:>10.0f} "
          f"{best_new[4]:>12.0f} {bottleneck(best_new[2], best_new[3]):>16}")
    print(f"  {'Oracle':<12} {best_actual[0]:>12.0f} {'—':>10} "
          f"{best_actual[4]:>12.0f} {bottleneck(best_actual[5], best_actual[6]):>16}")

    # Show top 10 candidates for new model
    hybrid_new_sorted = sorted(hybrid_new, key=lambda x: x[1])
    print(f"\n  Top 10 split thresholds (new model, predicted wall):")
    print(f"  {'Threshold':>10} {'Pred wall':>10} {'Pred CPU':>10} {'Pred GPU':>10} "
          f"{'Act wall':>10} {'Act CPU':>10} {'Act GPU':>10}")
    for thresh, wall_p, cpu_p, gpu_p, wall_a, cpu_a, gpu_a in hybrid_new_sorted[:10]:
        print(f"  {thresh:>10.0f} {wall_p:>10.0f} {cpu_p:>10.0f} {gpu_p:>10.0f} "
              f"{wall_a:>10.0f} {cpu_a:>10.0f} {gpu_a:>10.0f}")

    # =====================================================================
    # Phase 5: Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("Phase 5: Summary")
    print("=" * 70)

    print(f"\n  Model comparison:")
    print(f"  {'':>20} {'Old R²':>10} {'New R²':>10} {'Δ R²':>10}")
    print(f"  {'CPU standalone':<20} {r2_cpu_old:>10.4f} {r2_cpu_new:>10.4f} "
          f"{r2_cpu_new - r2_cpu_old:>+10.4f}")
    print(f"  {'GPU standalone':<20} {r2_gpu_old:>10.4f} {r2_gpu_new:>10.4f} "
          f"{r2_gpu_new - r2_gpu_old:>+10.4f}")

    actual_best_wall_old = min(h[4] for h in hybrid_old)
    actual_best_wall_new = min(h[4] for h in hybrid_new)
    # Find what split the old and new models recommend and what actual wall that gives
    old_rec_thresh = best_old[0]
    new_rec_thresh = best_new[0]
    # Find actual wall at old recommended threshold
    old_rec_actual = next((h[4] for h in hybrid_old if h[0] == old_rec_thresh), 0)
    new_rec_actual = next((h[4] for h in hybrid_new if h[0] == new_rec_thresh), 0)

    print(f"\n  Hybrid split quality:")
    print(f"    Old model recommends threshold {old_rec_thresh:.0f} → actual wall {old_rec_actual:.0f} µs")
    print(f"    New model recommends threshold {new_rec_thresh:.0f} → actual wall {new_rec_actual:.0f} µs")
    print(f"    Oracle (best actual):                     actual wall {best_actual[4]:.0f} µs")
    if old_rec_actual > 0:
        improvement = (old_rec_actual - new_rec_actual) / old_rec_actual * 100
        print(f"    New vs Old recommendation: {improvement:+.1f}%")

    print(f"\n  Recommended C++ constants:")
    print(f"    // CPU model: T_cpu = ALPHA_CPU * P * M * M")
    print(f"    const double ALPHA_CPU = {beta_cpu_old[0]:.6e};")
    print(f"")
    print(f"    // GPU model: T_gpu = GPU_GAMMA + GPU_ALPHA_M2 * M * M")
    print(f"    //                   + P * ceil(M/128) * M / SPEED_RATIO")
    gpu_speed = 1.0 / beta_gpu_new[2] if beta_gpu_new[2] > 0 else 75.0
    print(f"    const double GPU_GAMMA    = {beta_gpu_new[0]:.1f};     // µs fixed overhead")
    print(f"    const double GPU_ALPHA_M2 = {beta_gpu_new[1]:.6e};  // µs per M² (RMM setup)")
    print(f"    const double SPEED_RATIO  = {gpu_speed:.1f};  // 1/β₂ (GPU compute speed)")
    print(f"")
    print(f"    // To convert to PM²-equivalent units (for SPLIT_COST threshold):")
    print(f"    // GPU_GAMMA_PM2 = GPU_GAMMA / ALPHA_CPU = {beta_gpu_new[0] / beta_cpu_old[0]:.0f}")

    # =====================================================================
    # Phase 6: Optional sweep
    # =====================================================================
    if args.sweep:
        print("\n" + "=" * 70)
        print("Phase 6: Wall-time sweep validation")
        print("=" * 70)

        verbose_input2 = "_bench_v4.in"
        _make_verbose_local(args.input, verbose_input2)

        # Sweep SPLIT_COST values: all-GPU, recommended thresholds, all-CPU
        sweep_costs = sorted(set([
            0,  # all-GPU
            int(new_rec_thresh * 0.5),
            int(new_rec_thresh),
            int(new_rec_thresh * 2),
            int(old_rec_thresh),
            999999999999999,  # all-CPU
        ]))
        print(f"\n  {'SPLIT_COST':>14} {'Wall (s)':>10} {'CPU max':>10} {'GPU':>10}")
        for sc in sweep_costs:
            out, wall = run_liosolo(
                binary, verbose_input2, args.coords, args.basis,
                split_cost=sc)
            balance = parse_balance(out)
            avg_cpu = avg_gpu = 0
            if balance:
                avg_cpu = sum(c for c, g in balance) / len(balance)
                avg_gpu = sum(g for c, g in balance) / len(balance)
            print(f"  {sc:>14} {wall:>10.2f} {avg_cpu:>9.1f}ms {avg_gpu:>9.1f}ms")

        if os.path.exists(verbose_input2):
            os.remove(verbose_input2)


if __name__ == "__main__":
    main()
