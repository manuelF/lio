#!/usr/bin/env python3
"""
SPLITPOINTS auto-tuning benchmark for LIO CPU/GPU work distribution.

This utility measures per-group execution time on both CPU and GPU, fits
performance models, and recommends the optimal LIO_SPLIT_POINTS value for
the current hardware and molecular system.

Usage:
    cd test/LIO_test/03_fosfatoQMMM   # (or any test directory)
    python3 ../../../g2g/bench/splitpoint_tune.py \\
        --input fos.in --coords fos.xyz --basis basis

The script:
  1. Runs liosolo with LIO_SPLIT_POINTS=99999 (all-CPU) and verbose=4
  2. Runs liosolo with LIO_SPLIT_POINTS=1 (all-GPU) and verbose=4
  3. Parses [perfmodel] lines from stdout
  4. Fits T_cpu = α_cpu × P × M² and T_gpu = γ + α_gpu × P × M²
  5. Computes the crossover P×M² and optimal SPLITPOINTS
  6. Optionally sweeps SPLITPOINTS values to verify wall-time minimum

Requires: LIO built with `make cuda=1 cpu=1`, liohome.sh sourced.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import shutil


def find_liosolo():
    """Find liosolo binary, searching common locations."""
    candidates = [
        os.path.join(os.environ.get("LIOHOME", ""), "liosolo", "liosolo"),
        "liosolo",
        "../../../liosolo/liosolo",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    # Try PATH
    result = shutil.which("liosolo")
    if result:
        return result
    return None


def make_verbose_input(original_input, tmpdir):
    """Copy the input file, ensuring verbose=4 is set."""
    dst = os.path.join(tmpdir, os.path.basename(original_input))
    with open(original_input) as f:
        content = f.read()

    # Replace or add verbose setting
    if re.search(r'verbose\s*=', content, re.IGNORECASE):
        content = re.sub(r'verbose\s*=\s*\d+', 'verbose = 4', content,
                         flags=re.IGNORECASE)
    else:
        # Add verbose before &end
        content = re.sub(r'(&end)', r' verbose = 4\n\1', content,
                         flags=re.IGNORECASE)

    with open(dst, 'w') as f:
        f.write(content)
    return dst


def run_liosolo(binary, input_file, coords, basis, split_points, env_extra=None):
    """Run liosolo and return (stdout, wall_seconds)."""
    env = os.environ.copy()
    env["LIO_SPLIT_POINTS"] = str(split_points)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS",
                                      str(os.cpu_count() or 4))
    if env_extra:
        env.update(env_extra)

    cmd = [binary, "-i", input_file, "-c", coords, "-b", basis, "-v"]
    import time
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, env=env)
    except subprocess.TimeoutExpired:
        print("ERROR: liosolo timed out (600s limit)", file=sys.stderr)
        sys.exit(1)
    wall = time.monotonic() - t0
    return result.stdout + result.stderr, wall


def parse_perfmodel(output):
    """Parse [perfmodel] lines, return list of (points, functions, cost, time_us)."""
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if "[perfmodel]" not in line:
            continue
        line = line.replace("[perfmodel]", "").strip()
        if line.startswith("idx,"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append((
                int(parts[2]),    # points
                int(parts[3]),    # functions
                int(parts[4]),    # cost
                float(parts[5])   # time_us
            ))
        except (ValueError, IndexError):
            continue
    return rows


def fit_through_origin(xs, ys):
    """Fit y = a*x (no intercept). Returns (a, r²)."""
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    a = sxy / sxx if sxx > 0 else 0
    ss_res = sum((y - a * x) ** 2 for x, y in zip(xs, ys))
    mean_y = sum(ys) / len(ys) if ys else 0
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, r2


def fit_with_intercept(xs, ys):
    """Fit y = a*x + b. Returns (a, b, r²)."""
    n = len(xs)
    if n < 2:
        return 0, 0, 0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return 0, sy / n, 0
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, r2


def parse_balance(output):
    """Parse [balance] lines, return list of (cpu_ms, gpu_ms)."""
    rows = []
    for line in output.splitlines():
        m = re.search(r'\[balance\] CPU max=([\d.]+)ms.*GPU=([\d.]+)ms', line)
        if m:
            rows.append((float(m.group(1)), float(m.group(2))))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="LIO SPLITPOINTS auto-tuning benchmark")
    parser.add_argument("--input", "-i", required=True,
                        help="LIO input file (e.g. fos.in)")
    parser.add_argument("--coords", "-c", required=True,
                        help="Coordinate file (e.g. fos.xyz)")
    parser.add_argument("--basis", "-b", required=True,
                        help="Basis set file (e.g. basis)")
    parser.add_argument("--binary", default=None,
                        help="Path to liosolo binary (auto-detected if omitted)")
    parser.add_argument("--sweep", action="store_true",
                        help="Also sweep SPLITPOINTS values to verify prediction")
    parser.add_argument("--sweep-values", type=str, default=None,
                        help="Comma-separated SPLITPOINTS values to sweep "
                             "(default: auto-generated around crossover)")
    parser.add_argument("--skip-measure", action="store_true",
                        help="Skip measurement, read from --cpu-data and --gpu-data")
    parser.add_argument("--cpu-data", type=str, default=None,
                        help="File with all-CPU perfmodel output (skip CPU run)")
    parser.add_argument("--gpu-data", type=str, default=None,
                        help="File with all-GPU perfmodel output (skip GPU run)")
    parser.add_argument("--threads", type=int, default=None,
                        help="OMP_NUM_THREADS (default: auto-detect CPU count)")
    args = parser.parse_args()

    binary = args.binary or find_liosolo()
    if not binary:
        print("ERROR: Cannot find liosolo binary. Set --binary or source liohome.sh.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Using liosolo: {binary}")

    # Create temporary input file with verbose=4
    tmpdir = tempfile.mkdtemp(prefix="lio_bench_")
    verbose_input = make_verbose_input(args.input, tmpdir)

    cpu_data = []
    gpu_data = []

    if args.skip_measure:
        if args.cpu_data:
            with open(args.cpu_data) as f:
                cpu_data = parse_perfmodel(f.read())
        if args.gpu_data:
            with open(args.gpu_data) as f:
                gpu_data = parse_perfmodel(f.read())
    else:
        # Phase 1: All-CPU measurement
        print("\n" + "=" * 60)
        print("Phase 1: Measuring all groups on CPU (LIO_SPLIT_POINTS=99999)")
        print("=" * 60)
        omp_str = str(args.threads or os.cpu_count() or 4)
        cpu_out, cpu_wall = run_liosolo(
            binary, verbose_input, args.coords, args.basis,
            split_points=99999,
            env_extra={"OMP_NUM_THREADS": omp_str})
        cpu_data = parse_perfmodel(cpu_out)
        print(f"  Collected {len(cpu_data)} group timings, wall time: {cpu_wall:.1f}s")

        # Phase 2: All-GPU measurement
        print("\n" + "=" * 60)
        print("Phase 2: Measuring all groups on GPU (LIO_SPLIT_POINTS=1)")
        print("=" * 60)
        gpu_out, gpu_wall = run_liosolo(
            binary, verbose_input, args.coords, args.basis,
            split_points=1)
        gpu_data = parse_perfmodel(gpu_out)
        print(f"  Collected {len(gpu_data)} group timings, wall time: {gpu_wall:.1f}s")

    # Cleanup temp
    shutil.rmtree(tmpdir, ignore_errors=True)

    if not cpu_data or not gpu_data:
        print("ERROR: No perfmodel data collected. Is verbose=4 working?",
              file=sys.stderr)
        print("  Ensure LIO is built with cpu=1 cuda=1 and the [perfmodel] "
              "logging is present in partition.cpp.", file=sys.stderr)
        sys.exit(1)

    if len(cpu_data) != len(gpu_data):
        print(f"WARNING: CPU ({len(cpu_data)}) and GPU ({len(gpu_data)}) "
              f"group counts differ. Using min({len(cpu_data)}, {len(gpu_data)}).")

    n_groups = min(len(cpu_data), len(gpu_data))

    # Phase 3: Fit models
    print("\n" + "=" * 60)
    print("Phase 3: Fitting performance models")
    print("=" * 60)

    # CPU: T_cpu = α_cpu × P × M² (no intercept, negligible fixed overhead)
    cpu_filtered = [(p, m, c, t) for p, m, c, t in cpu_data if t > 5.0]
    cpu_x = [p * m * m for p, m, c, t in cpu_filtered]
    cpu_y = [t for p, m, c, t in cpu_filtered]

    if len(cpu_x) < 3:
        print("ERROR: Too few CPU data points for fitting.", file=sys.stderr)
        sys.exit(1)

    alpha_cpu, r2_cpu = fit_through_origin(cpu_x, cpu_y)

    print(f"\n  CPU model: T_cpu = α_cpu × P × M²")
    print(f"    α_cpu = {alpha_cpu:.6e} µs per unit P×M²")
    print(f"    R²    = {r2_cpu:.4f} (fitted on {len(cpu_filtered)} groups)")

    # GPU: T_gpu = γ + α_gpu × P × M² (skip first group = cold start)
    gpu_warm = [(p, m, c, t) for p, m, c, t in gpu_data[1:]]
    gpu_x = [p * m * m for p, m, c, t in gpu_warm]
    gpu_y = [t for p, m, c, t in gpu_warm]

    if len(gpu_x) < 3:
        print("ERROR: Too few GPU data points for fitting.", file=sys.stderr)
        sys.exit(1)

    alpha_gpu, gamma, r2_gpu = fit_with_intercept(gpu_x, gpu_y)

    print(f"\n  GPU model: T_gpu = γ + α_gpu × P × M²")
    print(f"    γ (fixed overhead) = {gamma:.1f} µs")
    print(f"    α_gpu = {alpha_gpu:.6e} µs per unit P×M²")
    print(f"    R²    = {r2_gpu:.4f} (fitted on {len(gpu_warm)} groups)")
    print(f"\n  GPU slope speedup: α_cpu / α_gpu = {alpha_cpu / alpha_gpu:.1f}×")

    # Phase 4: Crossover analysis
    print("\n" + "=" * 60)
    print("Phase 4: Crossover analysis")
    print("=" * 60)

    slope_diff = alpha_cpu - alpha_gpu
    if slope_diff <= 0:
        print("  WARNING: CPU is not faster per-unit than GPU (α_cpu <= α_gpu).")
        print("  GPU is faster for ALL group sizes. Set LIO_SPLIT_POINTS=1.")
        sys.exit(0)

    crossover_pm2 = gamma / slope_diff
    print(f"\n  Crossover P×M² = {crossover_pm2:.0f}")
    print(f"  Below this: CPU faster. Above: GPU faster.")

    # Compute per-group recommendations
    print(f"\n  Per-group analysis (N={n_groups} groups):")
    print(f"  {'Grp':>4} {'P':>6} {'M':>4} {'P×M²':>10} {'T_cpu µs':>9} "
          f"{'T_gpu µs':>9} {'Best':>5} {'Margin':>8}")

    cpu_better = 0
    gpu_better = 0
    misassigned_at_default = 0
    default_sp = 200

    for i in range(n_groups):
        p, m, _, t_cpu = cpu_data[i]
        _, _, _, t_gpu = gpu_data[i]
        pm2 = p * m * m
        best = "CPU" if t_cpu < t_gpu else "GPU"
        margin = abs(t_cpu - t_gpu) / max(t_cpu, t_gpu) * 100
        if best == "CPU":
            cpu_better += 1
        else:
            gpu_better += 1
        # Check if default SP=200 would assign correctly
        current_device = "GPU" if p > default_sp else "CPU"
        if current_device != best:
            misassigned_at_default += 1

        # Print groups near the crossover
        if abs(pm2 - crossover_pm2) < crossover_pm2 * 2:
            print(f"  {i:>4} {p:>6} {m:>4} {pm2:>10} {t_cpu:>9.1f} "
                  f"{t_gpu:>9.1f} {best:>5} {margin:>7.1f}%")

    print(f"\n  Summary: {cpu_better} groups faster on CPU, "
          f"{gpu_better} faster on GPU")
    print(f"  With default SPLITPOINTS={default_sp}: "
          f"{misassigned_at_default} groups misassigned")

    # Phase 5: Optimal SPLITPOINTS
    print("\n" + "=" * 60)
    print("Phase 5: Optimal SPLITPOINTS recommendation")
    print("=" * 60)

    # Collect the M values seen in the system to compute SP per typical M
    m_values = sorted(set(m for _, m, _, _ in cpu_data))
    median_m = m_values[len(m_values) // 2]
    min_m = m_values[0]
    max_m = m_values[-1]

    print(f"\n  Basis function range: M = {min_m}..{max_m} (median {median_m})")
    print(f"\n  SPLITPOINTS formula: SP(M) = {crossover_pm2:.0f} / M²")
    print(f"\n  {'M':>5} {'SP(M)':>8}")
    for m in sorted(set([min_m, median_m, max_m] + m_values[:3] + m_values[-3:])):
        sp = crossover_pm2 / (m * m)
        print(f"  {m:>5} {sp:>8.0f}")

    # The real metric is max(CPU_total, GPU_total) since they run in parallel.
    # CPU groups are distributed across cpu_threads; the bottleneck CPU thread
    # determines the CPU side. GPU groups run sequentially on 1 GPU thread.
    omp_threads = args.threads or int(os.environ.get(
        "OMP_NUM_THREADS", os.cpu_count() or 4))
    cpu_threads = max(1, omp_threads - 1)  # reserve 1 for GPU
    print(f"\n  Simulating parallel execution: {cpu_threads} CPU threads + 1 GPU thread")

    def simulate_parallel_time(sp, data_cpu, data_gpu, n_cpu_threads):
        """Simulate wall time = max(slowest_CPU_thread, GPU_thread).

        CPU groups are bin-packed using greedy LPT (Longest Processing Time)
        heuristic — same as LIO's compute_work_partition. This accounts for
        load imbalance from large groups that can't be split across threads.
        GPU groups run sequentially on 1 thread.
        """
        import heapq
        cpu_times_per_group = []
        gpu_total = 0.0
        for i in range(min(len(data_cpu), len(data_gpu))):
            p, m, _, t_cpu = data_cpu[i]
            _, _, _, t_gpu = data_gpu[i]
            if p <= sp:
                cpu_times_per_group.append(t_cpu)
            else:
                gpu_total += t_gpu

        if not cpu_times_per_group or n_cpu_threads < 1:
            return gpu_total, 0.0, gpu_total

        # LPT bin-packing: assign largest jobs first to least-loaded thread
        cpu_times_per_group.sort(reverse=True)
        # Min-heap of (load, thread_id)
        bins = [(0.0, i) for i in range(n_cpu_threads)]
        heapq.heapify(bins)
        for t in cpu_times_per_group:
            load, tid = heapq.heappop(bins)
            heapq.heappush(bins, (load + t, tid))

        cpu_bottleneck = max(load for load, _ in bins)
        return max(cpu_bottleneck, gpu_total), cpu_bottleneck, gpu_total

    # Test every actual P value as a potential SPLITPOINTS, plus some extras
    test_sps = sorted(set(
        [1, 50, 100, 200, 500, 1000, 2000, 3000, 5000] +
        [p for p, _, _, _ in cpu_data]
    ))

    best_sp = None
    best_wall = float('inf')
    best_cpu_ms = 0
    best_gpu_ms = 0
    all_results = []

    for sp in test_sps:
        if sp < 1:
            continue
        wall, cpu_ms, gpu_ms = simulate_parallel_time(
            sp, cpu_data, gpu_data, cpu_threads)
        all_results.append((sp, wall, cpu_ms, gpu_ms))
        if wall < best_wall:
            best_wall = wall
            best_sp = sp
            best_cpu_ms = cpu_ms
            best_gpu_ms = gpu_ms

    # Show top candidates
    all_results.sort(key=lambda x: x[1])
    print(f"\n  {'SP':>8} {'Wall µs':>10} {'CPU/thr µs':>12} {'GPU µs':>10} "
          f"{'Bottleneck':>12}")
    for sp, wall, cpu_ms, gpu_ms in all_results[:10]:
        bn = "CPU" if cpu_ms > gpu_ms else "GPU"
        marker = " <-- best" if sp == best_sp else ""
        print(f"  {sp:>8} {wall:>10.0f} {cpu_ms:>12.0f} {gpu_ms:>10.0f} "
              f"{bn:>12}{marker}")

    # Default SP=200 for comparison
    wall_200, cpu_200, gpu_200 = simulate_parallel_time(
        default_sp, cpu_data, gpu_data, cpu_threads)

    print(f"\n  Optimal SPLITPOINTS = {best_sp}")
    print(f"  Predicted iter time: {best_wall:.0f} µs "
          f"(vs {wall_200:.0f} µs at SP={default_sp})")
    idle_pct = abs(best_cpu_ms - best_gpu_ms) / best_wall * 100
    print(f"  CPU/thread: {best_cpu_ms:.0f} µs, GPU: {best_gpu_ms:.0f} µs, "
          f"idle: {idle_pct:.0f}%")
    improvement = (wall_200 - best_wall) / wall_200 * 100
    print(f"  Predicted improvement over default: {improvement:+.1f}%")
    print(f"\n  export LIO_SPLIT_POINTS={best_sp}")

    # Phase 6: Optional sweep
    if args.sweep:
        print("\n" + "=" * 60)
        print("Phase 6: SPLITPOINTS sweep (actual wall-time measurement)")
        print("=" * 60)

        if args.sweep_values:
            sweep_sps = [int(x) for x in args.sweep_values.split(",")]
        else:
            # Generate sweep values around the optimal
            sweep_sps = sorted(set([
                200,  # default
                best_sp,
                max(1, best_sp // 2),
                best_sp * 2,
                int(crossover_pm2 / (max_m * max_m)),
                int(crossover_pm2 / (min_m * min_m)),
            ]))

        print(f"\n  {'SP':>8} {'Wall (s)':>10} {'Note':>20}")
        results = []
        tmpdir2 = tempfile.mkdtemp(prefix="lio_sweep_")
        verbose_input2 = make_verbose_input(args.input, tmpdir2)

        for sp in sweep_sps:
            out, wall = run_liosolo(
                binary, verbose_input2, args.coords, args.basis,
                split_points=sp)
            balance = parse_balance(out)
            note = ""
            if sp == 200:
                note = "(default)"
            elif sp == best_sp:
                note = "(predicted optimal)"
            if balance:
                avg_cpu = sum(c for c, g in balance) / len(balance)
                avg_gpu = sum(g for c, g in balance) / len(balance)
                note += f" cpu={avg_cpu:.0f}ms gpu={avg_gpu:.0f}ms"
            print(f"  {sp:>8} {wall:>10.2f} {note:>20}")
            results.append((sp, wall))

        shutil.rmtree(tmpdir2, ignore_errors=True)

        best_sweep = min(results, key=lambda x: x[1])
        print(f"\n  Best measured: SP={best_sweep[0]}, wall={best_sweep[1]:.2f}s")

    # Final summary
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"\n  export LIO_SPLIT_POINTS={best_sp}")
    print(f"\n  Model parameters (for this system):")
    print(f"    α_cpu = {alpha_cpu:.6e} µs/(P×M²)")
    print(f"    α_gpu = {alpha_gpu:.6e} µs/(P×M²)")
    print(f"    γ     = {gamma:.1f} µs (GPU fixed overhead)")
    print(f"    Crossover P×M² = {crossover_pm2:.0f}")
    print(f"    GPU speedup (slope) = {alpha_cpu / alpha_gpu:.1f}×")


if __name__ == "__main__":
    main()
