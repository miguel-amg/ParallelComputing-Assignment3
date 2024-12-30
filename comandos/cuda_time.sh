#!/bin/sh
#SBATCH --exclusive
#SBATCH --time=05:00
#SBATCH --partition day
#SBATCH --constraint=k20
#SBATCH --ntasks=1

cd ..

module load gcc/7.2.0
module load cuda/11.3.1

make

time nvprof ./fluid_sim