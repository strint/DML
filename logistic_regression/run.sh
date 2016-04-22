#!/bin/bash

mpiexec -mca btl ^openib -np 2 ./train ./data/train.txt ./data/test_mini.txt
