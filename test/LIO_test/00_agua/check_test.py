#!/usr/bin/env python3
import sys

sys.path.insert(0,"../../tests_engine")
import energy
import fukui
import mulliken
import forces
import dipole

failed = False
failed |= bool(energy.Check())
failed |= bool(fukui.Check())
failed |= bool(mulliken.Check())
failed |= bool(forces.Check())
failed |= bool(dipole.Check())

sys.exit(1 if failed else 0)
