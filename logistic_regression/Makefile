#!/bin/bash
LIB = -L/opt/OpenBLAS/lib
INCLUDE = -I/opt/OpenBLAS/include 
#train code
train:main.o opt_algo.o load_data.o 
	mpicxx -o train main.o opt_algo.o load_data.o $(LIB)  -lopenblas -lpthread

main.o: src/main.cpp 
	mpicxx $(INCLUDE) -I ./include -c src/main.cpp

load_data.o: src/load_data.cpp
	mpicxx $(INCLUDE) -I ./include -c src/load_data.cpp

opt_algo.o: src/opt_algo.cpp  
	mpicxx $(INCLUDE) -I ./include -c src/opt_algo.cpp

#predict code
predict: predict.o
	mpicxx -g -o predict -lpthread $(LIB) -lopenblas predict.o

predict.o: src/predict.cpp
	mpicxx $(INCLUDE) -c src/predict.cpp

#make train uttest
train_ut: train_uttest.o opt_algo.o
	mpicxx -o train_ut train_uttest.o opt_algo.o $(LIB) -lopenblas -lpthread -L ./lib -lgtest

train_uttest.o: src/train_uttest.cpp
	mpicxx  -I ./include -c src/train_uttest.cpp
clean:
	rm -f *~ train predict train_ut *.o
