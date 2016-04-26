

CPPFLAGS = -O2
INCLUDES = -I/usr/include/mpi/include -I./include

.PHONY:all
all: mpi_eval

%.o : src/%.cpp
	 mpicxx $(CPPFLAGS) -c $< -o $@ $(INCLUDES)


mpi_eval: main.o eval.o load_data.o
	mpicxx $^ $(LDFLAGS) -o $@  $(INCLUDES)

.PHONY:clean
clean:
	rm -fr *.o
