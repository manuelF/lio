#!/usr/bin/env python3

import os
import subprocess
import itertools
from typing import List, Dict, Tuple, Iterator


def set_options(flag_set: Dict[str, str]) -> str:
    opt = ["%s=%s" % (k, v) for (k, v) in flag_set.items()]
    opt_str = "%s" % (" ".join(opt),)
    options = opt_str.rstrip()
    return options


def compile_lio(options: str) -> int:
    liodir = os.path.abspath("../")
    devnull = open(os.devnull, "w")
    cmd = ["tests_engine/build.sh", liodir, options]
    process = subprocess.Popen(cmd, stdout=devnull, stderr=devnull)
    process.wait()
    return process.returncode


def run_lio() -> int:
    cmd = ["./new_tests.py"]
    process = subprocess.Popen(cmd, cwd=os.path.abspath("."))
    process.wait()
    return process.returncode


def cuda_is_installed() -> bool:
    """Verifies if CUDA is installed."""
    devnull = open(os.devnull, "wb")
    command = ["nvcc --version"]

    process = subprocess.Popen(command, shell=True, stdout=devnull, stderr=devnull)
    try:
        _, _ = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    is_installed = retcode == 0
    return is_installed


def intel_is_installed() -> bool:
    """Checks if Intel compilers are present"""
    devnull = open(os.devnull, "wb")
    command = ["icc --version"]
    process = subprocess.Popen(command, shell=True, stdout=devnull, stderr=devnull)
    try:
        _, _ = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    is_installed = retcode == 0
    return is_installed


if __name__ == "__main__":
    comp = ["precision"]

    if cuda_is_installed():
        comp.append("cuda")
    if intel_is_installed():
        comp.append("intel")

    switches = ["0", "1"]
    all_combinations_tuple: Iterator[Tuple[str, ...]] = itertools.product(
        switches, repeat=len(comp)
    )
    all_combinations: List[Tuple[str, ...]] = list(all_combinations_tuple)
    all_sets: List[Dict[str, str]] = []

    for cases in all_combinations:
        compile_opts = dict(zip(comp, cases))
        if cuda_is_installed():
            if compile_opts["cuda"] == "1":
                compile_opts["cuda"] = "2"

        if intel_is_installed():
            if compile_opts["intel"] == "1":
                compile_opts["intel"] = "2"

        all_sets.append(compile_opts)

    for flag_set in all_sets:
        opts = set_options(flag_set)
        if not cuda_is_installed():
            opts = opts + " cuda=0 "
        print("Compiling LIO with Options: %s" % opts.rstrip())
        error = compile_lio(opts)
        if not error:
            print("\tSuccessfully compiled.")
        else:
            print("\tError!")

        error = run_lio()
        if not error:
            print("run_lio successfully finished.")
        else:
            print("run_lio Error!")
            exit(-1)
