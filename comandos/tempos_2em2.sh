#!/bin/bash
#SBATCH --exclusive      
#SBATCH --time=02:00
#SBATCH --partition=cpar 

cd ..
cd src

module load gcc/11.2.0 
make par

for threads in {20..40}; do
    echo "Running with OMP_NUM_THREADS=$threads"
    export OMP_NUM_THREADS=$threads
    time ./fluid_sim_par
done

make clean