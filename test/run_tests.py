#!/usr/bin/env python3
"""Run all LIO tests: unit tests first, then end-to-end integration tests.

This is the top-level test runner. It delegates to run_unit.py and run_e2e.py.

Usage:
    ./run_tests.py                          # run everything
    ./run_tests.py --filter_rx "energy"     # filter both unit and e2e tests
    ./run_tests.py --unit-only              # run only unit tests
    ./run_tests.py --e2e-only               # run only e2e tests
    ./run_tests.py --list                   # list all tests without running
    ./run_tests.py --no-build               # skip unit test build step
"""

import sys
import os
import argparse
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    """Import a sibling script as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Run all LIO tests.")
    parser.add_argument("--filter_rx", default=".*",
                        help="Regex to filter test names (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List all tests without running them")
    parser.add_argument("--unit-only", action="store_true",
                        help="Run only unit tests")
    parser.add_argument("--e2e-only", action="store_true",
                        help="Run only end-to-end tests")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip the unit test build step")
    parser.add_argument("--sanitize", action="store_true",
                        help="Run GPU unit tests under compute-sanitizer")
    args = parser.parse_args()

    run_unit = not args.e2e_only
    run_e2e = not args.unit_only

    unit_mod = load_module("run_unit", os.path.join(SCRIPT_DIR, "run_unit.py"))
    e2e_mod = load_module("run_e2e", os.path.join(SCRIPT_DIR, "run_e2e.py"))

    unit_passed, unit_failed, unit_skipped = [], [], []
    e2e_passed, e2e_failed, e2e_skipped = [], [], []

    # --- Unit tests ---
    if run_unit:
        unit_tests = unit_mod.discover_tests(args.filter_rx)
        if args.list:
            gpu = [t for t in unit_tests if t[1] == "gpu"]
            cpu = [t for t in unit_tests if t[1] == "cpu"]
            print(f"GPU kernel tests ({len(gpu)}):")
            for name, _, _ in gpu:
                print(f"  {name}")
            print(f"\nCPU conformance tests ({len(cpu)}):")
            for name, _, _ in cpu:
                print(f"  {name}")
        elif unit_tests:
            if not args.no_build:
                if not unit_mod.build(unit_tests):
                    print("Unit test build failed, skipping unit tests.\n")
                    unit_tests = []
            if unit_tests:
                unit_passed, unit_failed, unit_skipped = unit_mod.run_tests(
                    unit_tests, sanitize=args.sanitize)
                unit_mod.print_summary(unit_passed, unit_failed, unit_skipped)

    # --- E2E tests ---
    if run_e2e:
        e2e_tests = e2e_mod.discover_tests(args.filter_rx)
        if args.list:
            if run_unit:
                print()
            print(f"E2E integration tests ({len(e2e_tests)}):")
            for name, _ in e2e_tests:
                print(f"  {name}")
        elif e2e_tests:
            env = e2e_mod.lio_env()
            e2e_passed, e2e_failed, e2e_skipped = e2e_mod.run_tests(e2e_tests, env)
            e2e_mod.print_summary(e2e_passed, e2e_failed, e2e_skipped)

    if args.list:
        return 0

    # --- Grand summary ---
    total_passed = len(unit_passed) + len(e2e_passed)
    total_failed = len(unit_failed) + len(e2e_failed)
    total_skipped = len(unit_skipped) + len(e2e_skipped)
    total = total_passed + total_failed + total_skipped

    if run_unit and run_e2e:
        print(f"{'#'*60}")
        print(f"  Total: {total_passed} passed, {total_failed} failed, "
              f"{total_skipped} skipped out of {total}")
        print(f"{'#'*60}")

    all_failed = unit_failed + e2e_failed
    if all_failed:
        for name in all_failed:
            print(f"  FAILED: {name}")
        print()

    return 1 if all_failed else 0


if __name__ == "__main__":
    sys.exit(main())
