# Entry point for all CUDA kernel unit tests.
#
# Usage (from this directory):
#   make            - build and run all tests
#   make build      - compile all tests
#   make run        - run all already-compiled tests
#   make clean      - remove all build artifacts
#
# Optional overrides forwarded to every sub-Makefile:
#   CUDA_HOME=/path/to/cuda
#   CUDA_ARCH="-gencode arch=compute_80,code=sm_80"

SUITES := kernels

.PHONY: all build run clean $(SUITES)

all: $(SUITES)

$(SUITES):
	$(MAKE) -C $@ $(MAKECMDGOALS)

build run clean:
	@for s in $(SUITES); do \
	  $(MAKE) -C $$s $@; \
	done
