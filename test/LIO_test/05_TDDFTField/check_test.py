#!/usr/bin/env python3
import sys

sys.path.insert(0,"../../tests_engine")
import dipole
import restart

failed = False
failed |= bool(dipole.Check("td"))
failed |= bool(restart.Check())

sys.exit(1 if failed else 0)
