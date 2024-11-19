CPP = g++ -Wall -fopenmp -Ofast
SRCS = src/main.cpp src/fluid_solver.cpp src/EventManager.cpp

all: phase2

phase2:
	$(CPP) $(SRCS) -o fluid_sim
	cp src/events.txt ./ 

clean:
	@echo Cleaning up...
	@rm -f fluid_sim
	@echo Done.

runseq: all
	export OMP_NUM_THREADS=1 && \
	./fluid_sim

runpar: all
	export OMP_NUM_THREADS=21 && \
	./fluid_sim

# Nota: Events.txt é copiado para a diretoria do executavel para não ocorrer o erro "Error opening file: events.txt" 