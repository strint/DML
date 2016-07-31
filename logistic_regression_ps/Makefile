#!/bin/bash
CPP = g++
CPP_tag = -std=c++11
INCLUDE = -I../repo/ps-lite/include

#lr_ftrl: lr_main.o ftrl.o ../repo/ps-lite/build/libps.a
#	$(CPP) $(CPP_tag) -o lr_ftrl lr_main.o ftrl.o ../repo/ps-lite/build/libps.a -L../repo/ps-lite/deps/lib/ -lprotobuf-lite -lzmq

lr_main.o: src/lr_main.cpp ../repo/ps-lite/build/libps.a
	$(CPP) -c $(CPP_tag) src/lr_main.cpp $(INCLUDE)

#ftrl.o: src/ftrl.cpp ../repo/ps-lite/build/libps.a
#	$(CPP) -c $(CPP_tag) src/ftrl.cpp $(INCLUDE)

clean:
	rm -f *~ train
	rm -f *.o
