#!/bin/bash

mpiexec -mca btl ^openib -np 1 ./train ./data/train.txt ./data/test_mini.txt
