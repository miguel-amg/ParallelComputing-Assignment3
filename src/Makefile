CPP = g++ -Wall -Ofast -funroll-loops -ftree-vectorize -mtune=native -march=native -fopenmp -g -fno-omit-frame-pointer
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
