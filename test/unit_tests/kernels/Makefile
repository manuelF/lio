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
CXXFLAGS  := -std=c++14 -I$(COMMON_DIR) -I$(G2G_DIR) -O2 $(EXTRA_FLAGS)
LDFLAGS   := -lcudart

# ---------------------------------------------------------------------------
# Auto-discover GPU (*_test.cu) and CPU (*_cpu_test.cpp) test sources.
# ---------------------------------------------------------------------------
SRCS_CU  := $(wildcard *_test.cu)
BINS_CU  := $(SRCS_CU:.cu=)

SRCS_CPP := $(wildcard *_cpu_test.cpp)
BINS_CPP := $(SRCS_CPP:.cpp=)

BINS := $(BINS_CU) $(BINS_CPP)
CHECKS:= racecheck memcheck initcheck synccheck

.PHONY: all build run clean

all: build run

build: $(BINS)

%_test: %_test.cu $(COMMON_DIR)/test_utils.h $(COMMON_DIR)/kernels_reference.h
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

%_cpu_test: %_cpu_test.cpp $(COMMON_DIR)/cpu_test_utils.h $(COMMON_DIR)/kernels_reference.h \
            $(G2G_DIR)/cpu/cpu_kernels.h
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

run: build
	@for t in $(BINS); do \
	  for c in $(CHECKS); do \
	  echo "----------------------------------------"; \
	  echo "Running $$t ..."; \
	  echo "----------------------------------------"; \
	  /usr/local/cuda-13.1/bin/compute-sanitizer --tool $$c ./$$t; \
	done; \
	done

clean:
	rm -f $(BINS) *.o
