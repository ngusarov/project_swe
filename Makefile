# New MPICXX
MPICXX=mpicxx # Or mpiCC, or your system's MPI C++ compiler wrapper
CXX=${MPICXX} # Use MPICXX as the C++ compiler
LD=${MPICXX}  # Use MPICXX as the linker

# CXXFLAGS_COMMON = -g -O3 -Wall -Wextra -Werror -pedantic -std=c++11
CXXFLAGS_COMMON = -g -O3 -std=c++11
# Consider if -fno-inline is still needed. For initial MPI debugging it can be useful,
# but for performance later, you might want to remove it.
CXXFLAGS_COMMON += -fno-inline

# Include paths
CXXFLAGS += ${CXXFLAGS_COMMON} -I${HDF5_ROOT}/include

# Linker flags - ensure HDF5 libs are linked after MPI has a chance to link its own.
# $(CXXFLAGS_COMMON) might not be needed in LDFLAGS if it contains only compiler options.
# Typically LDFLAGS includes library paths and library names.
LDFLAGS_LIBS = -L${HDF5_ROOT}/lib -lhdf5 -lm


OBJS= main.o swe.o xdmf_writer.o

all: swe

swe: $(OBJS)
# Use $(LD) which is now $(MPICXX). Add common flags if they contain linker directives,
# otherwise, just library paths and names.
	$(LD) -o $@ $(OBJS) $(LDFLAGS_LIBS) $(CXXFLAGS_COMMON) # $(CXXFLAGS_COMMON) here for -g, -O3 etc. during link

# Rule for .cc to .o
%.o: %.cc %.hh swe.hh # Add swe.hh as a common dependency for .o files if it's widely included
	$(CXX) $(CXXFLAGS) -c $< -o $@

# For main.o, if it doesn't include xdmf_writer.hh directly
main.o: main.cc swe.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@

swe.o: swe.cc swe.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@

xdmf_writer.o: xdmf_writer.cc xdmf_writer.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f swe *.o *~
