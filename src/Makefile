CPP = g++ -Wall -fopenmp
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all: phase2

phase2:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid
	@echo Done.

runseq:
	./fluid_sim_seq

runpar:
	./fluid_sim