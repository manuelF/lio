#!/usr/bin/env python3
import sys

sys.path.insert(0,"../../tests_engine")
import energy
import forces

failed = False
failed |= bool(energy.Check())
failed |= bool(forces.Check())

sys.exit(1 if failed else 0)
