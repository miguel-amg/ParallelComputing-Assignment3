#!/bin/bash
#SBATCH --partition=cpar
#SBATCH --exclusive       # exclusive node for the job
#SBATCH --time=02:00      # allocation for 2 minutes
#SBATCH --cpus-per-task=40

cd ..

echo "A executar a versao sequencial"
time make runseq

echo "A executar a versao paralela"
time make runpar