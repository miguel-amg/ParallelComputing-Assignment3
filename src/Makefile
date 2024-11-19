CPP = g++ -Wall -fopenmp -Ofast
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all: phase2

phase2:
	module load gcc/11.2.0 && $(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim
	@echo Done.

runseq:
	export OMP_NUM_THREADS=1 && \
	./fluid_sim

runpar:
	export OMP_NUM_THREADS=21 && \
	./fluid_sim