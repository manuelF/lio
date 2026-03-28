################################################################################
# LIO MAKEFILE ################################################################################

all: liosolo tools

# Compilation phases — no cross-dependencies, run in parallel with -j.
# lioamber .o files don't need libg2g.so; only the link step does.
.PHONY: g2g-compile lioamber-compile
g2g-compile:
	$(MAKE) -C g2g compile

lioamber-compile:
	$(MAKE) -C lioamber compile

# g2g link — needs only g2g objects
.PHONY: g2g
g2g: g2g-compile
	$(MAKE) -C g2g link

# lioamber link — needs its objects AND libg2g.so
.PHONY: liblio
liblio: lioamber-compile g2g
	$(MAKE) -C lioamber link

# liosolo — needs both .so files
.PHONY: liosolo
liosolo: liblio
	$(MAKE) -C liosolo

# Convenience aliases for backward compatibility
.PHONY: libg2g
libg2g: g2g

.PHONY: tools
tools:
	$(MAKE) -C tools

.PHONY: check check-unit check-e2e
check:
	$(MAKE) check -C test
check-unit:
	$(MAKE) check-unit -C test
check-e2e:
	$(MAKE) check-e2e -C test

.PHONY: compile
compile:
	$(MAKE) compile -C test

.PHONY: clean
clean:
	$(MAKE) clean -C liosolo
	$(MAKE) clean -C lioamber
	$(MAKE) clean -C g2g
	$(MAKE) clean -C test
	$(MAKE) clean -C tools

################################################################################
