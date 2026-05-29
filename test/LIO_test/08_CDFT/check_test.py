#!/usr/bin/env python3
import sys

sys.path.insert(0,"../../tests_engine")
import mulliken
import becke

failed = False
failed |= bool(mulliken.Check())
failed |= bool(becke.Check())

sys.exit(1 if failed else 0)
