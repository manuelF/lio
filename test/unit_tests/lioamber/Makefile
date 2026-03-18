# Makefile for lioamber Fortran unit tests
#
# Prerequisites: lioamber must be built first (make liblio from root).
# Tests link against pre-compiled objects from lioamber/obj/.
#
# Usage:
#   make          - build and run all tests
#   make build    - compile only
#   make run      - run already-compiled tests
#   make clean    - remove build artifacts

FC       ?= gfortran
OBJDIR    = ../../../lioamber/obj
G2GDIR    = ../../../g2g
LIODIR    = ../../../lioamber

FFLAGS    = -cpp -I$(OBJDIR) -J$(OBJDIR) -fPIC -fno-omit-frame-pointer -Dpack -DG2G -DTD_SIMPLE -O2
LFLAGS    = -L$(G2GDIR) -Wl,-rpath=$(realpath $(G2GDIR))
LIBS      = -lg2g -lblas -llapack

# ---------------------------------------------------------------------------
# Test programs and their object dependencies
# ---------------------------------------------------------------------------

# test-properties: standalone (includes properties.f90 directly)
PROPERTIES_OBJS =

# test-mathsubs: needs mathsubs.o + linear_algebra.o (for matmuldiag)
MATHSUBS_OBJS = $(OBJDIR)/mathsubs.o \
                $(OBJDIR)/linear_algebra.o

# test-converger: needs operator type + converger + linear_algebra
CONVERGER_OBJS = $(OBJDIR)/converger_subs.o \
                 $(OBJDIR)/converger_data.o \
                 $(OBJDIR)/typedef_operator.o \
                 $(OBJDIR)/fileio.o \
                 $(OBJDIR)/linear_algebra.o \
                 $(OBJDIR)/mathsubs.o

# test-propagators: needs propagators + large dependency chain
PROPAGATORS_OBJS = $(OBJDIR)/propagators.o \
                   $(OBJDIR)/garcha_mod.o \
                   $(OBJDIR)/basis_data.o \
                   $(OBJDIR)/constants_mod.o \
                   $(OBJDIR)/mathsubs.o \
                   $(OBJDIR)/fileio.o \
                   $(OBJDIR)/linear_algebra.o \
                   $(OBJDIR)/fockbias_subs.o \
                   $(OBJDIR)/fockbias_data.o \
                   $(OBJDIR)/subm_int3lu.o \
                   $(OBJDIR)/field_subs.o \
                   $(OBJDIR)/faint_cpu.o \
                   $(OBJDIR)/liosubs.o \
                   $(OBJDIR)/liosubs_math.o \
                   $(OBJDIR)/typedef_operator.o \
                   $(OBJDIR)/mask_ecp.o \
                   $(OBJDIR)/mask_cublas.o \
                   $(OBJDIR)/shape_data.o \
                   $(OBJDIR)/packed_storage.o \
                   $(OBJDIR)/dip.o \
                   $(OBJDIR)/subm_intfld.o \
                   $(OBJDIR)/ECP_mod.o \
                   $(OBJDIR)/intECP.o \
                   $(OBJDIR)/esp_funct.o \
                   $(OBJDIR)/readECP.o

# test-int3lu: needs subm_int3lu.o + basis_data.o + fileio.o
INT3LU_OBJS = $(OBJDIR)/subm_int3lu.o \
              $(OBJDIR)/basis_data.o \
              $(OBJDIR)/fileio.o

# test-scf-blas: needs fock_commuts.o + BLAS
SCF_BLAS_OBJS = $(OBJDIR)/fock_commuts.o

TESTS = test-properties.x test-mathsubs.x test-converger.x test-propagators.x test-int3lu.x test-scf-blas.x

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all build run clean

all: build run

build: $(TESTS)

run:
	@passed=0; failed=0; errors=""; \
	for t in $(TESTS); do \
	  if [ ! -f $$t ]; then \
	    echo "SKIP: $$t (not built)"; \
	    continue; \
	  fi; \
	  echo "--- Running $$t ---"; \
	  ./$$t; \
	  if [ $$? -eq 0 ]; then \
	    passed=$$((passed + 1)); \
	  else \
	    failed=$$((failed + 1)); \
	    errors="$$errors $$t"; \
	  fi; \
	done; \
	echo ""; \
	echo "=== Results: $$passed passed, $$failed failed ==="; \
	if [ $$failed -gt 0 ]; then \
	  echo "Failed:$$errors"; \
	  exit 1; \
	fi

test-properties.x: test-properties.f90
	$(FC) $(FFLAGS) -I$(LIODIR) $< $(PROPERTIES_OBJS) -o $@

test-mathsubs.x: test-mathsubs.f90
	$(FC) $(FFLAGS) $(LFLAGS) $< $(MATHSUBS_OBJS) $(LIBS) -o $@

test-converger.x: test-converger.f90
	$(FC) $(FFLAGS) $(LFLAGS) $< $(CONVERGER_OBJS) $(LIBS) -o $@

test-propagators.x: test-propagators.f90
	$(FC) $(FFLAGS) $(LFLAGS) $< $(PROPAGATORS_OBJS) $(LIBS) -o $@

test-int3lu.x: test-int3lu.f90
	$(FC) $(FFLAGS) $(LFLAGS) $< $(INT3LU_OBJS) $(LIBS) -o $@

test-scf-blas.x: test-scf-blas.f90
	$(FC) $(FFLAGS) $(LFLAGS) $< $(SCF_BLAS_OBJS) $(LIBS) -o $@

clean:
	rm -f *.x *.mod
