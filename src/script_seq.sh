#!/bin/sh
#
#SBATCH --exclusive
#SBATCH --time=02:00
 
perf stat -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores -o perf_output_seq.txt -- ./fluid_sim_seq