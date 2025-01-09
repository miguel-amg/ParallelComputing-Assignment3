# Configuração do compilador e flags
CXX = nvcc
FLAGS = --compiler-options -Wall -O3 -g -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets -Xcompiler -fopenmp
SRCS = src/main.cpp src/fluid_solver.cu src/EventManager.cpp
EXEC = fluid_sim

# Regra principal
all: phase3

# Regra phase3
phase3:
	module load gcc/7.2.0; \
	module load cuda/11.3.1; \
	$(CXX) $(FLAGS) $(SRCS) -o $(EXEC)

# Submissão SLURM
run: 
	sbatch --exclusive --time=05:00 --partition=day --constraint=k20 --ntasks=1 --wrap "\
		module load gcc/7.2.0; \
		module load cuda/11.3.1; \
		nvprof ./$(EXEC)"

# Limpar os ficheiros
clean:
	@echo Cleaning up...
	@rm -f $(EXEC)
	@echo Done.

# Execução com 1 thread
runseq: 
	export OMP_NUM_THREADS=1 && ./$(EXEC)

# Execução com 21 threads
runpar: 
	export OMP_NUM_THREADS=21 && ./$(EXEC)
