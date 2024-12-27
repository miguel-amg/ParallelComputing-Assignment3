CXX = nvcc
FLAGS = --compiler-options -Wall -O3 -g -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets -Xcompiler -fopenmp
SRCS = src/main.cpp src/fluid_solver.cu src/EventManager.cpp

all: phase3

phase3:
	$(CXX) $(FLAGS) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.

runseq: 
	export OMP_NUM_THREADS=1 && \
	./fluid_sim

runpar: 
	export OMP_NUM_THREADS=21 && \
	./fluid_sim

# Nota: Events.txt é copiado para a diretoria do executavel para não ocorrer o erro "Error opening file: events.txt"
