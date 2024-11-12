CPP = g++ -Wall -O3 -Ofast -pg -fopenmp
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
TARGET= ../fluid_sim

all:
	$(CPP) $(SRCS) -o fluid_sim

assembly:
	$(CPP) -S $(SRCS)
	
clean:
	@echo Cleaning up...
	@rm -f ../fluid_sim
	@echo Done.
