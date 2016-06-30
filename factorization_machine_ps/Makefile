#!/bin/bash
include ./lib/ps-lite/make/deps.mk
all: build/difacto.dmlc build/dump.dmlc

clean:
	rm -rf build *.pb.*

build/difacto.dmlc: build/config.pb.o build/difacto.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@

build/dump.dmlc: build/dump.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@

#LIB=/home/services/xiaoshu/lib
#INCLUDE=/home/services/xiaoshu/include
#train code
#train:fm_main.o ftrl.o load_data.o predict.o
#	/home/xiaoshu/hadoop_job/bin/mpicxx  -o train fm_main.o ftrl.o load_data.o predict.o  -lpthread

#fm_main.o: src/fm_main.cpp 
#	/home/xiaoshu/hadoop_job/bin/mpicxx -I ./include -c src/fm_main.cpp

#load_data.o: src/load_data.cpp
#	/home/xiaoshu/hadoop_job/bin/mpicxx -I ./include -c src/load_data.cpp

#ftrl.o: src/ftrl.cpp
#	/home/xiaoshu/hadoop_job/bin/mpicxx -I ./include -c src/ftrl.cpp

#predict.o: src/predict.cpp
#	/home/xiaoshu/hadoop_job/bin/mpicxx -I ./include -c src/predict.cpp

#clean:
#	rm -f *~ train predict *.o
