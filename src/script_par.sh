#!/bin/sh
#
#SBATCH --exclusive
#SBATCH --time=02:00
 
for threads in 16 17 18 19 20
do
    export OMP_NUM_THREADS=$threads
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
    perf stat -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores -o perf_output_${threads}.txt -- ./fluid_sim_par
    echo
done