#!/bin/bash
mpiexec -mca btl ^openib -np 1 ./train ./data/agaricus.txt.train ./data/agaricus.txt.train
