CPP = g++
FLAGS_SEQ = -Wall -Ofast -funroll-loops -ftree-vectorize -mtune=native -march=native
FLAGS_PAR = -Wall -Ofast -funroll-loops -ftree-vectorize -mtune=native -march=native -fopenmp
SRCS_PAR = main.cpp fluid_solver_par.cpp EventManager.cpp
SRCS_SEQ = main.cpp fluid_solver_seq.cpp EventManager.cpp

all: seq par

seq:
	$(CPP) $(FLAGS_SEQ) $(SRCS_SEQ) -o fluid_sim_seq

par:
	$(CPP) $(FLAGS_PAR) $(SRCS_PAR) -o fluid_sim_par

clean:
	@echo Cleaning up...
	@rm -f fluid_sim_par fluid_sim_seq
	@echo Done.

runseq:
	./fluid_sim_seq

runpar:
	./fluid_sim_par