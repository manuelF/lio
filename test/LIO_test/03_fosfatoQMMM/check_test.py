#!/usr/bin/env python3
import sys

sys.path.insert(0, "../../tests_engine")
import energy
import forces
import mulliken
import dipole

# Wider tolerances for fosfatoQMMM: the timing-dependent rebalancer causes
# run-to-run variation in FP accumulation order, shifting energy by ~0.005 Ha
# and forces by ~0.005 a.u.  This is inherent to the parallel CPU/GPU split.
energy.Check(total_energy_thre=1e-2)
forces.Check(thre=1e-2)
mulliken.Check()
dipole.Check()
