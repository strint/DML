#!/bin/bash
#include ./lib/ps-lite/make/deps.mk
#all: build/difacto.dmlc build/dump.dmlc

#clean:
#	rm -rf build *.pb.*

#build/difacto.dmlc: build/config.pb.o build/difacto.o $(DMLC_SLIB)
#	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@

#build/dump.dmlc: build/dump.o $(DMLC_SLIB)
#	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@

CC = g++

#fm:fm_main.o ftrl.o predict.o ./lib/ps-lite/build/libps.a
#	$(CC)  -o train fm_main.o ftrl.o predict.o  ./lib/ps-lite/build/libps.a -L./ps-lite/deps/lib/ -lprotobuf-lite -lzmq -std=c++11 -lpthread

#fm_main.o: src/fm_main.cpp include/load_data.h 
#	$(CC) -c src/fm_main.cpp -I./include -I./lib/ps-lite/include -std=c++11

ftrl.o: src/ftrl.cpp
	$(CC) -c src/ftrl.cpp -I./include -I./lib/ps-lite/include -std=c++11

clean:
	rm -f train
	rm -f *.o
