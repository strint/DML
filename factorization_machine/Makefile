#!/bin/bash
LIB = -L/opt/OpenBLAS/lib
INCLUDE = -I/opt/OpenBLAS/include 
GLOG_LIB = -L/usr/local/lib
GLOG_INCLUDE = -I/usr/local/include/glog
GTEST_LIB = -L/usr/local/lib
GTEST_INCLUDE = -I/usr/local/include/gtest
#train code
CPP_tag = -std=gnu++11
train:fm_main.o ftrl.o load_data.o predict.o
	mpicxx $(CPP_tag) -o train fm_main.o ftrl.o load_data.o predict.o $(LIB) $(GLOG_LIB) -lopenblas -lpthread -lglog

fm_main.o: src/fm_main.cpp 
	mpicxx $(CPP_tag) $(INCLUDE) $(GLOG_INCLUDE) -I ./include -c src/fm_main.cpp

load_data.o: src/load_data.cpp
	mpicxx $(CPP_tag) $(INCLUDE) $(GLOG_INCLUDE) -I ./include -c src/load_data.cpp

ftrl.o: src/ftrl.cpp
	mpicxx $(CPP_tag) $(INCLUDE) $(GLOG_INCLUDE) -I ./include -c src/ftrl.cpp

predict.o: src/predict.cpp
	mpicxx $(CPP_tag) $(INCLUDE) $(GLOG_INCLUDE) -I ./include -c src/predict.cpp

clean:
	rm -f *~ train predict train_ut *.o
