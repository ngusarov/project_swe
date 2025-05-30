# Makefile

# New MPICXX
MPICXX=mpicxx # Or mpiCC, or your system's MPI C++ compiler wrapper
CXX=${MPICXX} # Use MPICXX as the C++ compiler
LD=${MPICXX}  # Use MPICXX as the linker

# CXXFLAGS_COMMON = -g -O3 -Wall -Wextra -Werror -pedantic -std=c++11
# CXXFLAGS_COMMON = -g -O3 -std=c++11
CXXFLAGS_COMMON = -g -Wall -std=c++11

# Consider if -fno-inline is still needed. For initial MPI debugging it can be useful,
# but for performance later, you might want to remove it.
CXXFLAGS_COMMON += -fno-inline

# Include paths
CXXFLAGS += ${CXXFLAGS_COMMON} -I${HDF5_ROOT}/include

# Linker flags - ensure HDF5 libs are linked after MPI has a chance to link its own.
# $(CXXFLAGS_COMMON) might not be needed in LDFLAGS if it contains only compiler options.
# Typically LDFLAGS includes library paths and library names.
LDFLAGS_LIBS = -L${HDF5_ROOT}/lib -lhdf5 -lhdf5_hl -lm


OBJS= main.o swe.o xdmf_writer.o

all: swe

swe: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS_LIBS) $(CXXFLAGS_COMMON) # <--- This line MUST start with a TAB

# Rule for .cc to .o
# Add xdmf_writer.hh as a dependency for swe.o
%.o: %.cc %.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@ # <--- This line MUST start with a TAB

# Explicit rules for clarity, ensuring correct dependencies
main.o: main.cc swe.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@ # <--- This line MUST start with a TAB

swe.o: swe.cc swe.hh xdmf_writer.hh # IMPORTANT: Dependency added for xdmf_writer.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@ # <--- This line MUST start with a TAB

xdmf_writer.o: xdmf_writer.cc xdmf_writer.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@ # <--- This line MUST start with a TAB

clean:
	rm -f swe *.o *~ # <--- This line MUST start with a TAB
