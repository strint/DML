#!/bin/bash
CPP = g++
CPP_tag = -std=c++11
DEPS_PATH=../repo/ps-lite/deps
INCLUDE = -I../repo/ps-lite/src -I../repo/ps-lite/deps/include

LDFLAGS = $(addprefix $(DEPS_PATH)/lib/, libglog.a libprotobuf.a libgflags.a libzmq.a libcityhash.a liblz4.a)


all: lr_ftrl

lr_ftrl: lr_main.o ../repo/ps-lite/build/libps.a $(LDFLAGS) -lpthread
	$(CPP) $(CPP_tag) -o $@ $^ $(INCLUDE)

lr_main.o: src/lr_main.cpp ../repo/ps-lite/build/libps.a
	$(CPP) $(CPP_tag) -c src/lr_main.cpp $(INCLUDE)

clean:
	rm -f *~ train
	rm -f *.o
