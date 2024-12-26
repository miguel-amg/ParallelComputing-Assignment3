#!/bin/sh
#SBATCH --exclusive
#SBATCH --time=02:00
#SBATCH --partition day
#SBATCH --cpus-per-task=48
#SBATCH --constraint=c24
#SBATCH --ntasks=1

cd ..

module load gcc/11.2.0 
make

export OMP_NUM_THREADS=21
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
perf stat -r 3 -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores -- ./fluid_sim
