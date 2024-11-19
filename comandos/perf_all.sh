#!/bin/bash
#SBATCH --exclusive
#SBATCH --partition=cpar
#SBATCH --cpus-per-task=40

cd ..
cd src

module load gcc/11.2.0 
make

for threads in {1..40}; do
    export OMP_NUM_THREADS=$threads
    echo "Executando com OMP_NUM_THREADS=$threads e cpus-per-task=40"

    perf stat -e instructions,cycles,duration_time ./fluid_sim

    echo "Finalizado para OMP_NUM_THREADS=$threads"
    echo "--------------------------------------------"
done