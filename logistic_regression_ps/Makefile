#!/bin/bash
#train code
CPP = g++
CPP_tag = -std=gnu++11
INCLUDE = -I./include -I../lib/ps-lite/include

#lr_ftrl: lr_main.o ftrl.o load_data.o
#	$(CPP) $(CPP_tag) -I./include -I../lib/ps-lite/include -o train lr_main.o ftrl.o

#lr_main.o: src/lr_main.cpp 
#	$(CPP) $(CPP_tag) -c src/lr_main.cpp $(INCLUDE)

ftrl.o: src/ftrl.cpp
	$(CPP) $(CPP_tag) -c src/ftrl.cpp $(INCLUDE)

clean:
	rm -f *~ train
	rm -f *.o
