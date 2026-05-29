#!/usr/bin/env python3
import sys

sys.path.insert(0,"../../tests_engine")
import energy

failed = False
failed |= bool(energy.Check())

sys.exit(1 if failed else 0)
