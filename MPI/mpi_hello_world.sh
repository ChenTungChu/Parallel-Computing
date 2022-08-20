#!/bin/bash

# module load openmpi-4.0.1
# mpirun -np 6 /bin/hostname
mpirun -np 12 ./mpi_hello_world --mca opal_warn_on_missing_libcuda 0

