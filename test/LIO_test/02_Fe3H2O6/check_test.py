#!/usr/bin/env python3
import sys

sys.path.insert(0,"../../tests_engine")
import restart
import energy

failed = False
failed |= bool(restart.Check())
failed |= bool(energy.Check())

sys.exit(1 if failed else 0)
