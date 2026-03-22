COMMON_DIR := $(abspath ../common)
G2G_DIR    := $(abspath ../../../g2g)

NVCC      ?= nvcc
CXX       ?= g++
CUDA_HOME ?= /usr/local/cuda

# ---------------------------------------------------------------------------
# GPU architecture detection — mirrors g2g/Makefile.cuda exactly.
# ---------------------------------------------------------------------------
ifeq ($(GENCODE_FLAGS),)
  DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap \
                   --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
  ifneq ($(DETECTED_SM),)
    GENCODE_FLAGS := -gencode arch=compute_$(DETECTED_SM),code=compute_$(DETECTED_SM)
    GENCODE_FLAGS += -gencode arch=compute_$(DETECTED_SM),code=sm_$(DETECTED_SM)
  endif
endif

CUDA_VER  := $(shell nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
CUDA_OLD  := $(shell [ -n "$(CUDA_VER)" ] && [ "$(CUDA_VER)" -lt 8 ] && echo 1 || echo 0)
ifeq ($(CUDA_OLD),1)
  GENCODE_FLAGS ?= -gencode arch=compute_30,code=compute_30 \
                   -gencode arch=compute_30,code=sm_30
else
  GENCODE_FLAGS ?= -gencode arch=compute_50,code=compute_50 \
                   -gencode arch=compute_50,code=sm_50 \
                   -gencode arch=compute_60,code=compute_60 \
                   -gencode arch=compute_60,code=sm_60 \
                   -gencode arch=compute_75,code=compute_75 \
                   -gencode arch=compute_75,code=sm_75
endif

NVCCFLAGS := -std=c++14 -I$(COMMON_DIR) -I$(G2G_DIR) $(GENCODE_FLAGS) \
             -g -G -O0 --ptxas-options=-v $(EXTRA_FLAGS)
CXXFLAGS  := -std=c++14 -I$(COMMON_DIR) -I$(G2G_DIR) -g -O2 $(EXTRA_FLAGS)
LDFLAGS   := -lcudart

# ---------------------------------------------------------------------------
# Auto-discover GPU (*_test.cu) and CPU (*_cpu_test.cpp) test sources.
# ---------------------------------------------------------------------------
SRCS_CU  := $(wildcard *_test.cu)
BINS_CU  := $(SRCS_CU:.cu=)

SRCS_CPP := $(wildcard *_cpu_test.cpp)
BINS_CPP := $(SRCS_CPP:.cpp=)

BINS := $(BINS_CU) $(BINS_CPP)

# compute-sanitizer path: auto-detect from PATH, fall back to known locations.
SANITIZER := $(shell command -v compute-sanitizer 2>/dev/null)
ifeq ($(SANITIZER),)
  ifneq ($(wildcard $(CUDA_HOME)/bin/compute-sanitizer),)
    SANITIZER := $(CUDA_HOME)/bin/compute-sanitizer
  endif
endif

SANITIZER_CHECKS := memcheck racecheck initcheck synccheck

.PHONY: all build run sanitize clean

all: build run

build: $(BINS)

%_test: %_test.cu $(COMMON_DIR)/test_utils.h $(COMMON_DIR)/kernels_reference.h
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

%_cpu_test: %_cpu_test.cpp $(COMMON_DIR)/cpu_test_utils.h $(COMMON_DIR)/kernels_reference.h \
            $(G2G_DIR)/cpu/cpu_kernels.h
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

# Default run: execute each test directly (fast).
run: build
	@for t in $(BINS); do \
	  echo "----------------------------------------"; \
	  echo "Running $$t ..."; \
	  echo "----------------------------------------"; \
	  ./$$t; \
	done

# Sanitize: run each GPU test under compute-sanitizer with all check tools (slow).
sanitize: build
ifeq ($(SANITIZER),)
	$(error compute-sanitizer not found. Set CUDA_HOME or add it to PATH.)
endif
	@for t in $(BINS_CU); do \
	  for c in $(SANITIZER_CHECKS); do \
	    echo "========================================"; \
	    echo "Sanitizing $$t ($$c) ..."; \
	    echo "========================================"; \
	    $(SANITIZER) --tool $$c ./$$t; \
	  done; \
	done

clean:
	rm -f $(BINS) *.o
