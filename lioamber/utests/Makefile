######################################################################
# MAKEFILE FOR TESTS
######################################################################

FC = gfortran
FFLAGS = -fno-omit-frame-pointer
TESTS = test-properties.x test-ftermbias.x

all: $(TESTS)

%.x: %.f90
	$(FC) $(FFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf *.x *.out

######################################################################
