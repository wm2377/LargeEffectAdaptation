# Build the simulation core into the shared library simulation_cpp.py loads.
#
#   make            -- build ../libsimcore.so
#   make clean      -- remove it
#
# -march=native is deliberately NOT used: the sweep's jobs land on whatever node
# SLURM gives them, and a library built on a newer node would crash with an
# illegal instruction on an older one. -O2 plus -ffp-contract=off keeps the
# arithmetic identical across the nodes that do run it.

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fPIC -Wall -Wextra -ffp-contract=off
TARGET   := ../libsimcore.so
SOURCES  := sim_core.cpp
HEADERS  := rng.hpp sdist.hpp

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean
