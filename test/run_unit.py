#!/usr/bin/env python3
"""Run LIO unit tests (CUDA kernel tests + CPU conformance tests).

Test binaries live in test/unit_tests/kernels/ and are auto-built from
*_test.cu (GPU) and *_cpu_test.cpp (CPU) sources.

Usage:
    ./run_unit.py                       # build and run all unit tests
    ./run_unit.py --filter_rx "energy"  # run only tests matching regex
    ./run_unit.py --list                # list available tests without running
    ./run_unit.py --no-build            # skip build step, run existing binaries
    ./run_unit.py --sanitize            # run under compute-sanitizer (slow)
"""

import re
import os
import sys
import glob
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(SCRIPT_DIR, "unit_tests", "kernels")


def discover_tests(filter_rx=".*"):
    """Return sorted list of (test_name, kind, abs_binary_path)."""
    tests = []
    # GPU tests: *_test (from *_test.cu)
    for src in sorted(glob.glob(os.path.join(KERNELS_DIR, "*_test.cu"))):
        name = os.path.basename(src).replace(".cu", "")
        if re.search(filter_rx, name):
            tests.append((name, "gpu", os.path.join(KERNELS_DIR, name)))

    # CPU tests: *_cpu_test (from *_cpu_test.cpp)
    for src in sorted(glob.glob(os.path.join(KERNELS_DIR, "*_cpu_test.cpp"))):
        name = os.path.basename(src).replace(".cpp", "")
        if re.search(filter_rx, name):
            tests.append((name, "cpu", os.path.join(KERNELS_DIR, name)))

    return tests


def build(tests):
    """Build test binaries via make."""
    # Only build if there are tests to run.
    if not tests:
        return True
    print(f"Building unit tests in {KERNELS_DIR} ...")
    proc = subprocess.run(
        ["make", "build"], cwd=KERNELS_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if proc.returncode != 0:
        print("Build failed:")
        print(proc.stdout)
        return False
    print("Build OK.\n")
    return True


def run_tests(tests, sanitize=False):
    """Run each test binary, parse [PASS]/[FAIL] output. Return (passed, failed, skipped)."""
    passed = []
    failed = []
    skipped = []

    sanitizer_path = None
    if sanitize:
        # Try to find compute-sanitizer.
        for candidate in [
            "/usr/local/cuda/bin/compute-sanitizer",
            "/usr/local/cuda-13.1/bin/compute-sanitizer",
        ]:
            if os.path.isfile(candidate):
                sanitizer_path = candidate
                break
        if sanitizer_path:
            print(f"Using sanitizer: {sanitizer_path}\n")
        else:
            print("Warning: compute-sanitizer not found, running without it.\n")

    for name, kind, binary in tests:
        tag = "GPU" if kind == "gpu" else "CPU"
        print(f"{'='*60}")
        print(f"  Unit: {name} [{tag}]")
        print(f"{'='*60}")

        if not os.path.isfile(binary):
            print(f"  [SKIP] binary not found: {binary}")
            skipped.append(name)
            continue

        cmd = [binary]
        if sanitize and sanitizer_path and kind == "gpu":
            cmd = [sanitizer_path, "--tool", "memcheck", binary]

        proc = subprocess.run(
            cmd, cwd=KERNELS_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=120,
        )

        # Parse and display output.
        test_failed = False
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Show test structure lines.
            if stripped.startswith("===") or stripped.startswith("[OK"):
                print(f"  {stripped}")
            elif stripped.startswith("[PASS]"):
                print(f"  {stripped}")
            elif stripped.startswith("[FAIL]"):
                print(f"  {stripped}")
                test_failed = True
            elif stripped.startswith("["):
                # Other bracketed output (e.g. section headers).
                print(f"  {stripped}")

        if proc.returncode != 0:
            test_failed = True
            if not any("[FAIL]" in l for l in proc.stdout.splitlines()):
                print(f"  [FAIL] exit code {proc.returncode}")

        if test_failed:
            failed.append(name)
        else:
            passed.append(name)

    return passed, failed, skipped


def print_summary(passed, failed, skipped):
    total = len(passed) + len(failed) + len(skipped)
    print(f"\n{'='*60}")
    print(f"  Unit Summary: {len(passed)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped out of {total}")
    print(f"{'='*60}")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
    if skipped:
        for name in skipped:
            print(f"  SKIPPED: {name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run LIO unit tests.")
    parser.add_argument("--filter_rx", default=".*",
                        help="Regex to filter test names (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List tests without running them")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip the build step")
    parser.add_argument("--sanitize", action="store_true",
                        help="Run GPU tests under compute-sanitizer (slow)")
    args = parser.parse_args()

    tests = discover_tests(args.filter_rx)
    if not tests:
        print("No unit tests found.")
        return 1

    if args.list:
        gpu_tests = [t for t in tests if t[1] == "gpu"]
        cpu_tests = [t for t in tests if t[1] == "cpu"]
        print(f"GPU kernel tests ({len(gpu_tests)}):")
        for name, _, _ in gpu_tests:
            print(f"  {name}")
        print(f"\nCPU conformance tests ({len(cpu_tests)}):")
        for name, _, _ in cpu_tests:
            print(f"  {name}")
        return 0

    if not args.no_build:
        if not build(tests):
            return 1

    passed, failed, skipped = run_tests(tests, sanitize=args.sanitize)
    print_summary(passed, failed, skipped)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
