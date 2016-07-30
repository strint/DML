#!/bin/bash
#train code
CPP = g++
CPP_tag = -std=c++11
INCLUDE = -I../lib/ps-lite/include

lr_ftrl: lr_main.o ftrl.o ../lib/ps-lite/build/libps.a
	$(CPP) $(CPP_tag) -o lr_ftrl lr_main.o ftrl.o ../lib/ps-lite/build/libps.a -L../lib/ps-lite/deps/lib/ -lprotobuf-lite -lzmq

lr_main.o: src/lr_main.cpp ../lib/ps-lite/build/libps.a
	$(CPP) -c $(CPP_tag) src/lr_main.cpp $(INCLUDE)

ftrl.o: src/ftrl.cpp ../lib/ps-lite/build/libps.a
	$(CPP) -c $(CPP_tag) src/ftrl.cpp $(INCLUDE)

clean:
	rm -f *~ train
	rm -f *.o
