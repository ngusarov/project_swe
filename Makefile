CXX=g++
LD=${CXX}
CXXFLAGS+=-O3 -Wall -Wextra -Werror -pedantic -std=c++11 -I${HDF5_ROOT}/include
LDFLAGS+=-lm $(CXXFLAGS) -L${HDF5_ROOT}/lib -lhdf5

OBJS= main.o swe.o xdmf_writer.o

all: swe

swe: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f swe xdmf *.o *~
