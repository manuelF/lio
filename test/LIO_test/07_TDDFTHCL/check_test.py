#!/usr/bin/env python3
import sys

sys.path.insert(0,"../../tests_engine")
import dipole

failed = False
failed |= bool(dipole.Check("td"))

sys.exit(1 if failed else 0)
