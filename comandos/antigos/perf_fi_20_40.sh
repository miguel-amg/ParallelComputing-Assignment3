#!/bin/sh
#SBATCH --exclusive
#SBATCH --time=02:00
#SBATCH --partition=cpar 
#SBATCH --cpus-per-task=40

cd ..
cd src

module load gcc/11.2.0 
make par

for threads in {20..40}; do
    echo "Resultado para $threads"
    export OMP_NUM_THREADS=$threads
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
    perf stat -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores -- ./fluid_sim_par
    echo ---------------------------------------------------------------
done