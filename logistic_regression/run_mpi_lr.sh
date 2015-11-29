#!/bin/bash
mpiexec -mca btl ^openib -np 2 ./train 
