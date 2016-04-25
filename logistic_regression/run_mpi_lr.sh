#!/bin/bash

tt=`date`
mkdir backup/"$tt"
mv train backup/"$tt"
mv *.log backup/"$tt"
mv core backup/"$tt"
make
mpiexec -mca btl ^openib -np 4 ./train ./data/agaricus.txt.train ./data/agaricus.txt.train
