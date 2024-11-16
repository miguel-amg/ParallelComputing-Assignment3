#!/bin/sh
#
#SBATCH --exclusive
#SBATCH --time=02:00
 
export OMP_NUM_THREADS=61
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
time ./fluid_sim_par
echo
export OMP_NUM_THREADS=62
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
time ./fluid_sim_par
echo
export OMP_NUM_THREADS=63
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
time ./fluid_sim_par
echo
export OMP_NUM_THREADS=64
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
time ./fluid_sim_par
echo