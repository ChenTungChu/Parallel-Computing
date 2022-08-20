#!/bin/bash

# module load openmpi-4.0.1
# mpirun -np 12 /bin/hostname
mpirun -np 12 ./cc2396 --mca opal_warn_on_missing_libcuda 0