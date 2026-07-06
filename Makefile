# Master Makefile - builds every tier.
# Flags pass through automatically to each subdirectory, e.g.:
#   make PLATFORM=linux
#   make CXX=g++
#   make clean

SUBDIRS := 01uniquesoundgen 02chordlapsing

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
