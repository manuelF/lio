#!/usr/bin/env python3

import re
import os
import sys
import argparse
import subprocess

def lio_env():
   """"
   Sets lio enviroment variables, adding g2g and
   lioamber to LD_LIBRARY_PATH.
   """

   lioenv = os.environ.copy()
   lioenv["LIOBIN"] = os.path.abspath("../liosolo/liosolo")
   dirs = ["../g2g", "../lioamber"]
   try:
      prev = lioenv["LD_LIBRARY_PATH"]
   except:
      prev = ""
   lioenv["LD_LIBRARY_PATH"] = ":".join([prev] + [os.path.abspath(p) for p in dirs])
   lioenv["LIOHOME"] = os.path.abspath("../")
   return lioenv


def run_lio(dirs_with_tests):
   "Runs all Tests. Returns True if every test passed, False otherwise."
   lioenv = lio_env()
   all_passed = True

   for dir in dirs_with_tests:
      is_file = os.path.isfile(os.path.abspath(dir) + "/run.sh")
      print("Running LIO in",dir)

      if is_file:
        execpath = ["./run.sh"]
        process = subprocess.Popen(execpath, env=lioenv, cwd=os.path.abspath(dir))
        process.wait()
        if process.returncode != 0:
           print("Error running LIO in", dir)
           all_passed = False
           continue
        else:
           execpath = ["./check_test.py"]
           check = subprocess.Popen(execpath, env=lioenv, cwd=os.path.abspath(dir))
           check.wait()
           if check.returncode != 0:
              print("Test check FAILED in", dir)
              all_passed = False
      else:
        print("Nothing to do.")

   return all_passed


# Curated subset of tests that run on a CPU-only (cuda=0) build in CI.
# Selected by name -- never by positional index -- so adding or renaming a
# test folder cannot silently change which tests run.
CPU_TESTS = [
   "00_agua",
   "01_OxyMol",
   "03_fosfatoQMMM",
   "04_2_ECP",
   "06_QMinPcharges",
]

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--filter_rx", help="RegExp used to filter which tests are run.", default=".*")
   args = parser.parse_args()
   filterrx = args.filter_rx

   # Build the list of test folders from the explicit allowlist, keeping only
   # those that exist on disk and match the optional regex filter.
   subdirs = set(list(os.walk('LIO_test/'))[0][1])
   missing = [d for d in CPU_TESTS if d not in subdirs]
   if missing:
      print("\033[1;31mError! CPU test folders missing:", missing, "\033[0m")
      sys.exit(1)
   dirs_with_tests = ["LIO_test/" + d for d in sorted(CPU_TESTS)
                      if re.search(filterrx, d)]
   if not dirs_with_tests:
      print("No tests matched filter:", filterrx)
      sys.exit(0)

   # Run lio
   all_passed = run_lio(dirs_with_tests)
   if not all_passed:
      print("\033[1;31mOne or more tests FAILED.\033[0m")
      sys.exit(1)
   print("\033[1;32mAll tests passed.\033[0m")
   sys.exit(0)

