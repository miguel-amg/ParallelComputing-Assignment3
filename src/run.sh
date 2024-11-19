#!/bin/sh
#
#SBATCH --exclusive
#SBATCH --time=02:00

export OMP_NUM_THREADS=1
perf stat -e cycles,instructions,branch-misses,L1-dcache-loads,L1-dcache-load-misses,mem-loads,mem-stores ./fluid_sim
echo