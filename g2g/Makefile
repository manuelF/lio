.DEFAULT_GOAL := all

# -*- mode: make -*-
include Makefile.translate
include Makefile.common

## Define sources and objects
SRCS:=$(wildcard *.cpp)
OBJ:=$(SRCS:%.cpp=%.o)

POINTXC_SRC  := $(wildcard pointxc/*.cpp)
SRCS         += $(POINTXC_SRC)
OBJ          += $(POINTXC_SRC:%.cpp=%.o)

AINT_SRC     := $(wildcard analytic_integral/*.cpp)
SRCS         += $(AINT_SRC)
OBJ          += $(AINT_SRC:%.cpp=%.o)

CPU_SOURCES  := $(wildcard cpu/*.cpp)
SRCS         += $(CPU_SOURCES)
OBJ          += $(CPU_SOURCES:%.cpp=%.o)
CXXFLAGS     += -DCPU_KERNELS=1 -I$(MKLROOT)/include -Wno-unused-function
CPU          := 1

# Define LIBXC CPU/GPU
LIBXC_CPU_LIBS=$(LIBXC_HOME_CPU)/lib
LIBXC_GPU_LIBS=$(LIBXC_HOME_GPU)/lib
LIBXC_CPU_INCLUDES=$(LIBXC_HOME_CPU)/include
LIBXC_GPU_INCLUDES=$(LIBXC_HOME_GPU)/include

ifneq ($(cuda),0)
  include Makefile.cuda
  CUDA_LIBS    := -lcudart -lcuda -lcublas
  SRCS         += $(CUDA_SOURCES)
  OBJ          += $(UGO) $(CUDA_OBJ)
  CXXFLAGS     += -DGPU_KERNELS=1
  NVCCFLAGS    += -DGPU_KERNELS=1
endif

ifeq ($(libxc),1)
  LIBXC_LIBRARIES := -L$(LIBXC_CPU_LIBS) -lxc
#LIBXC CPU MODE
  CXXFLAGS += -DUSE_LIBXC=1 -DLIBXC_CPU=1 -I$(LIBXC_CPU_INCLUDES)
else
ifeq ($(libxc),2)
#LIBXC GPU MODE
    LIBXC_LIBRARIES := -L$(LIBXC_GPU_LIBS) -lxc_cuda -L$(LIBXC_CPU_LIBS) -lxc
    CXXFLAGS += -DUSE_LIBXC=1 -DLIBXC_CPU=0 -I$(LIBXC_GPU_INCLUDES) -I$(LIBXC_CPU_INCLUDES)
else
#NO LIBXC -- do not link libxc at all (the fake stubs are compiled instead).
    LIBXC_LIBRARIES :=
    CXXFLAGS += -DUSE_LIBXC=0 -DLIBXC_CPU=1
endif
endif

ifeq ($(aint_mp),1)
  CXXFLAGS     += -DAINT_MP=1
  NVCCFLAGS    += -DAINT_MP=1
endif

## Other options
ifeq ($(print), 1)
        CXXFLAGS += -DPRINT_MATRICES
endif

ifeq ($(profile),1)
  CXXFLAGS += -pg -g
  LDFLAGS += -pg -g
endif

ifeq ($(cpu_recompute),1)
  CXXFLAGS += -DCPU_RECOMPUTE=1
else
  CXXFLAGS += -DCPU_RECOMPUTE=0
endif

ifeq ($(full_double),1)
  CXXFLAGS += -DFULL_DOUBLE=1
  NVCCFLAGS += -DFULL_DOUBLE=1
else
  CXXFLAGS += -DFULL_DOUBLE=0
  NVCCFLAGS += -DFULL_DOUBLE=0
endif

## Define libraries
LIBRARIES := $(CUDA_LDFLAGS) $(CUDA_LIBS) $(LIBXC_LIBRARIES) -lrt
ifeq ($(intel),0)
  # Only link cblas from PACK_LIB_DIR when that path is actually provided.
  # Historically PACK_LIB_DIR was unset, which left a bare `-L` that swallowed
  # the following `-lcblas` token as its directory argument -- so cblas was
  # never really linked into libg2g.so (its symbols resolve at the final
  # executable link against -lopenblas). Make that explicit instead of relying
  # on the accidental drop; set PACK_LIB_DIR to link a standalone cblas.
  LIBRARIES += $(if $(PACK_LIB_DIR),-L$(PACK_LIB_DIR) -lcblas)
  CXXFLAGS += -DINTEL_COMP=0
else 
  CXXFLAGS += -DINTEL_COMP=1
endif


LIBRARY=libg2g.so
## Targets
all: $(LIBRARY)

ifeq ($(libint),1)
    LIBS_LIBINT = -L/$(LIBINT_HOME)/lib -lint2
    INCS_LIBINT = -I/$(LIBINT_HOME)/include -I/$(LIBINT_HOME)/include/libint2 -I/$(EIGEN_HOME)
    LIBRARIES  += $(LIBS_LIBINT)
    CXXFLAGS   += -std=c++11 $(INCS_LIBINT) -DUSE_LIBINT=1

    SRCS       += libint/libintproxy.cpp libint/g2g_libint.cpp
    OBJ        += libint/libintproxy.o libint/g2g_libint.o
else
    SRCS       += libint/g2g_libint_fake.cpp
    OBJ        += libint/g2g_libint_fake.o
    CXXFLAGS   += -DUSE_LIBINT=0
endif
include Makefile.excited
include Makefile.cioverlap

# Pull in the auto-generated header dependencies (.d files emitted by -MMD -MP).
# Declared after every Makefile.* include so $(OBJ) is fully populated. The
# leading '-' silences the missing-file warnings on a clean tree.
-include $(OBJ:.o=.d)

# Split targets so the top-level Makefile can overlap g2g and lioamber
# object compilation under `make -j`. Declared after every Makefile.* include
# so $(OBJ) is fully populated (libint, excited, cioverlap, ...).
.PHONY: all compile link clean
compile: $(OBJ)
link: $(LIBRARY)

# --version-script limits exports to the bridge API (libg2g.map); -Wl,-O1
# optimizes the symbol hash table. No --as-needed: this link has no
# --no-undefined guard, so a dropped-but-needed lib would only fail at runtime.
libg2g.so: $(OBJ) libg2g.map
	$(CXX) -fopenmp -shared $(LDFLAGS) -Wl,-O1 -Wl,--version-script=libg2g.map -o libg2g.so $(OBJ) $(LIBRARIES)

clean:
	@echo "Removing objects"; rm -f *.o* *.d libg2g.so *.a cpu/*.o* cpu/*.d cuda/*.o* cuda/*.d pointxc/*.o* pointxc/*.d
	@rm -f cuda/*.cu_o cuda/*.cudafe* cuda/*.ptx cuda/*.hash cuda/*.cubin cuda/*.i cuda/*.ii cuda/*.fatbin.* cuda/*.cu.c
	@rm -f analytic_integral/*.o* analytic_integral/*.d analytic_integral/cuda/*.o* analytic_integral/cuda/*.d
	@rm -f excited/*.o* excited/*.d excited/cuda/*.o* excited/cuda/*.d libint/*o libint/*.d
