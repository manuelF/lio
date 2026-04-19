#!/usr/bin/env python3
"""Run LIO unit tests (CUDA kernel tests + CPU conformance tests).

Test binaries live in test/unit_tests/kernels/ and are auto-built from
*_test.cu (GPU) and *_cpu_test.cpp (CPU) sources.

Usage:
    ./run_unit.py                       # build and run all unit tests
    ./run_unit.py --filter_rx "energy"  # run only tests matching regex
    ./run_unit.py --list                # list available tests without running
    ./run_unit.py --no-build            # skip build step, run existing binaries
    ./run_unit.py --sanitize            # run all sanitizer tools (slow)
    ./run_unit.py --sanitize=racecheck  # run specific sanitizer tool only
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


SANITIZER_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def find_sanitizer():
    """Locate compute-sanitizer (CUDA 13.1 moved it out of bin/)."""
    for candidate in [
        "/usr/local/cuda/bin/compute-sanitizer",
        "/usr/local/cuda-13.1/bin/compute-sanitizer",
        "/usr/local/cuda-13.1/compute-sanitizer/compute-sanitizer",
        "/usr/bin/compute-sanitizer",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def run_tests(tests, sanitize=None):
    """Run each test binary, parse [PASS]/[FAIL] output.

    sanitize: None (no sanitizer), "all" (run every tool), or a specific tool
    name from SANITIZER_TOOLS. Each tool runs as a separate pass; a hazard in
    any tool marks the test failed.
    """
    passed = []
    failed = []
    skipped = []

    sanitizer_path = None
    sanitizer_tools = []
    if sanitize:
        sanitizer_path = find_sanitizer()
        if sanitizer_path:
            if sanitize == "all":
                sanitizer_tools = list(SANITIZER_TOOLS)
            elif sanitize in SANITIZER_TOOLS:
                sanitizer_tools = [sanitize]
            else:
                print(f"Unknown sanitizer tool: {sanitize}. "
                      f"Valid: {', '.join(SANITIZER_TOOLS)} or 'all'.")
                return passed, failed, skipped
            print(f"Using sanitizer: {sanitizer_path}")
            print(f"Tools: {', '.join(sanitizer_tools)}\n")
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

        # Build list of passes: either one plain run, or one per sanitizer tool.
        if sanitizer_path and sanitizer_tools and kind == "gpu":
            passes = [(tool, [sanitizer_path, "--tool", tool, binary])
                      for tool in sanitizer_tools]
        else:
            passes = [(None, [binary])]

        test_failed = False
        for tool_name, cmd in passes:
            if tool_name:
                print(f"  --- sanitizer: {tool_name} ---")
            try:
                proc = subprocess.run(
                    cmd, cwd=KERNELS_DIR,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    timeout=300 if tool_name else 120,
                )
            except subprocess.TimeoutExpired:
                print(f"  [FAIL] {tool_name or 'run'} timed out")
                test_failed = True
                continue

            # Parse test output.
            sanitizer_errors = 0
            sanitizer_warnings = 0
            for line in proc.stdout.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                # Sanitizer summary line, e.g.
                # "========= RACECHECK SUMMARY: 67 hazards displayed (67 errors, 6 warnings)"
                # Must check this BEFORE the "===" prefix branch below.
                if "SUMMARY:" in stripped and "=========" in stripped:
                    m = re.search(r"\((\d+)\s+errors?,\s*(\d+)\s+warnings?\)",
                                  stripped)
                    if m:
                        sanitizer_errors = int(m.group(1))
                        sanitizer_warnings = int(m.group(2))
                    print(f"  {stripped}")
                elif stripped.startswith("===") or stripped.startswith("[OK"):
                    print(f"  {stripped}")
                elif stripped.startswith("[PASS]"):
                    print(f"  {stripped}")
                elif stripped.startswith("[FAIL]"):
                    print(f"  {stripped}")
                    test_failed = True
                elif stripped.startswith("["):
                    print(f"  {stripped}")

            if tool_name:
                ok_str = "ok" if sanitizer_errors == 0 else "FAIL"
                print(f"  [{ok_str}] {tool_name}: "
                      f"{sanitizer_errors} errors, {sanitizer_warnings} warnings")
                if sanitizer_errors > 0:
                    test_failed = True

            if proc.returncode != 0:
                test_failed = True
                if not any("[FAIL]" in l for l in proc.stdout.splitlines()):
                    print(f"  [FAIL] {tool_name or 'run'}: exit {proc.returncode}")

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
    parser.add_argument(
        "--sanitize", nargs="?", const="all", default=None,
        help="Run GPU tests under compute-sanitizer. With no value runs all "
             "tools (memcheck, racecheck, initcheck, synccheck). Pass a tool "
             "name (e.g. --sanitize=racecheck) to run just that one. Slow.")
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
    if failed and args.sanitize:
        print("\nNOTE: sanitizer hazards count as test failures. Run without "
              "--sanitize to confirm functional correctness, and inspect raw "
              "compute-sanitizer output for details.")
    print_summary(passed, failed, skipped)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
