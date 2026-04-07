# Makefile for regenerating Obara-Saika term files from OS_expand.plx
#
# Usage:
#   make              # regenerate all term files
#   make qmmm         # regenerate only QM/MM terms (2-index)
#   make coulomb      # regenerate only Coulomb terms (3-index)
#   make forces       # regenerate only force terms (gradient=1)
#   make energy       # regenerate only energy terms (gradient=0)
#   make clean        # remove all generated files (use with caution)
#
# The generator takes: perl OS_expand.plx <gradient> <L1> <L2> [<L3>]
#   gradient: 0 = energy, 1 = forces
#   L1, L2:   angular momentum of QM basis pair (0=s, 1=p, 2=d)
#   L3:       angular momentum of fitting function (Coulomb only)

GENERATOR := perl OS_expand.plx

KERNELS := cuda/kernels

# QM/MM term types: L1 >= L2 (upper triangle)
# ss=0,0  ps=1,0  pp=1,1  ds=2,0  dp=2,1  dd=2,2
QMMM_TERMS := ss ps pp ds dp dd

# Coulomb fitting functions
FIT_TYPES := s p d

# Map term names to L1,L2 arguments
ss_ARGS := 0 0
ps_ARGS := 1 0
pp_ARGS := 1 1
ds_ARGS := 2 0
dp_ARGS := 2 1
dd_ARGS := 2 2

# Map fitting function to L3 argument
s_L3 := 0
p_L3 := 1
d_L3 := 2

# --- Targets ---

.PHONY: all qmmm coulomb forces energy clean

all: qmmm coulomb

qmmm: qmmm-energy qmmm-forces
coulomb: coulomb-energy coulomb-forces
forces: qmmm-forces coulomb-forces
energy: qmmm-energy coulomb-energy

.PHONY: qmmm-energy qmmm-forces coulomb-energy coulomb-forces

# QM/MM energy terms (2-index, gradient=0)
qmmm-energy:
	@echo "Generating QM/MM energy terms..."
	@$(foreach t,$(QMMM_TERMS), \
		$(GENERATOR) 0 $($(t)_ARGS) > $(KERNELS)/qmmm_terms/energy/$(t).h ; )
	@echo "  Generated $(words $(QMMM_TERMS)) files in $(KERNELS)/qmmm_terms/energy/"

# QM/MM force terms (2-index, gradient=1)
qmmm-forces:
	@echo "Generating QM/MM force terms..."
	@$(foreach t,$(QMMM_TERMS), \
		$(GENERATOR) 1 $($(t)_ARGS) > $(KERNELS)/qmmm_terms/forces/$(t).h ; )
	@echo "  Generated $(words $(QMMM_TERMS)) files in $(KERNELS)/qmmm_terms/forces/"

# Coulomb energy terms (3-index, gradient=0)
coulomb-energy:
	@echo "Generating Coulomb energy terms..."
	@$(foreach t,$(QMMM_TERMS), \
		$(foreach f,$(FIT_TYPES), \
			$(GENERATOR) 0 $($(t)_ARGS) $($(f)_L3) > $(KERNELS)/coulomb_terms/energy/$(t)_$(f).h ; ))
	@echo "  Generated $(shell echo '$(words $(QMMM_TERMS)) * $(words $(FIT_TYPES))' | bc) files in $(KERNELS)/coulomb_terms/energy/"

# Coulomb force terms (3-index, gradient=1)
coulomb-forces:
	@echo "Generating Coulomb force terms..."
	@$(foreach t,$(QMMM_TERMS), \
		$(foreach f,$(FIT_TYPES), \
			$(GENERATOR) 1 $($(t)_ARGS) $($(f)_L3) > $(KERNELS)/coulomb_terms/forces/$(t)_$(f).h ; ))
	@echo "  Generated $(shell echo '$(words $(QMMM_TERMS)) * $(words $(FIT_TYPES))' | bc) files in $(KERNELS)/coulomb_terms/forces/"

clean:
	rm -f $(foreach t,$(QMMM_TERMS), $(KERNELS)/qmmm_terms/energy/$(t).h)
	rm -f $(foreach t,$(QMMM_TERMS), $(KERNELS)/qmmm_terms/forces/$(t).h)
	rm -f $(foreach t,$(QMMM_TERMS), $(foreach f,$(FIT_TYPES), $(KERNELS)/coulomb_terms/energy/$(t)_$(f).h))
	rm -f $(foreach t,$(QMMM_TERMS), $(foreach f,$(FIT_TYPES), $(KERNELS)/coulomb_terms/forces/$(t)_$(f).h))
