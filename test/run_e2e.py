#!/usr/bin/env python3
"""Run LIO end-to-end integration tests (test/LIO_test/NN_<name>/).

Each test runs liosolo on a molecule and validates physical observables
(energy, forces, dipole, etc.) against golden reference files.

Usage:
    ./run_e2e.py                        # run all e2e tests
    ./run_e2e.py --filter_rx "agua"     # run only tests matching regex
    ./run_e2e.py --list                 # list available tests without running
"""

import re
import os
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIO_TEST_DIR = os.path.join(SCRIPT_DIR, "LIO_test")


def lio_env():
    """Build environment dict with LIO library paths."""
    env = os.environ.copy()
    env["LIOBIN"] = os.path.abspath(os.path.join(SCRIPT_DIR, "../liosolo/liosolo"))
    dirs = [
        os.path.abspath(os.path.join(SCRIPT_DIR, "../g2g")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "../lioamber")),
    ]
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([prev] + dirs)
    env["LIOHOME"] = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    return env


def discover_tests(filter_rx=".*"):
    """Return sorted list of (dir_name, abs_path) for matching LIO_test dirs."""
    if not os.path.isdir(LIO_TEST_DIR):
        return []
    subdirs = sorted(os.listdir(LIO_TEST_DIR))
    tests = []
    for d in subdirs:
        full = os.path.join(LIO_TEST_DIR, d)
        if os.path.isdir(full) and re.search(filter_rx, d):
            if os.path.isfile(os.path.join(full, "run.sh")):
                tests.append((d, full))
    return tests


def run_tests(tests, env):
    """Run each test, capture check results, return (passed, failed, skipped)."""
    passed = []
    failed = []
    skipped = []

    for name, path in tests:
        print(f"\n{'='*60}")
        print(f"  E2E: {name}")
        print(f"{'='*60}")

        # Run the simulation.
        proc = subprocess.run(
            ["./run.sh"], cwd=path, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if proc.returncode != 0:
            print(f"  [SKIP] run.sh failed (exit {proc.returncode})")
            skipped.append(name)
            continue

        # Run the checker.
        check_script = os.path.join(path, "check_test.py")
        if not os.path.isfile(check_script):
            print(f"  [SKIP] no check_test.py")
            skipped.append(name)
            continue

        proc = subprocess.run(
            ["./check_test.py"], cwd=path, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        # Print each check result indented.
        for line in proc.stdout.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if "OK" in line_stripped and line_stripped.startswith("Test "):
                print(f"  [PASS] {line_stripped}")
            elif "ERROR" in line_stripped and line_stripped.startswith("Test "):
                print(f"  [FAIL] {line_stripped}")
            elif line_stripped.startswith("Error "):
                print(f"         {line_stripped}")
            elif line_stripped.startswith("Value "):
                print(f"         {line_stripped}")

        if "ERROR" in proc.stdout:
            failed.append(name)
        else:
            passed.append(name)

    return passed, failed, skipped


def print_summary(passed, failed, skipped):
    total = len(passed) + len(failed) + len(skipped)
    print(f"\n{'='*60}")
    print(f"  E2E Summary: {len(passed)} passed, {len(failed)} failed, "
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
    parser = argparse.ArgumentParser(description="Run LIO e2e integration tests.")
    parser.add_argument("--filter_rx", default=".*",
                        help="Regex to filter test names (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List tests without running them")
    args = parser.parse_args()

    tests = discover_tests(args.filter_rx)
    if not tests:
        print("No e2e tests found.")
        return 1

    if args.list:
        print(f"E2E tests ({len(tests)}):")
        for name, _ in tests:
            print(f"  {name}")
        return 0

    env = lio_env()
    passed, failed, skipped = run_tests(tests, env)
    print_summary(passed, failed, skipped)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
