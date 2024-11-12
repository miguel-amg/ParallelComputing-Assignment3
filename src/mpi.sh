#!/bin/bash
#SBATCH --partition=cpar
#SBATCH --exclusive

perf record -g ./fluid_sim
perf report --stdio>result
