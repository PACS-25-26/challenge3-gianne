# ---------------------------------------------------------------
# Makefile for the Laplace Solver (MPI + OpenMP, C++17)
# ---------------------------------------------------------------
# Usage:
#   make          -> build the executable
#   make clean    -> remove compiled files
#   make run      -> build and run with default parameters (4 procs)
#   make test     -> run the scalability script
# ---------------------------------------------------------------

# Compiler and flags
CXX       = mpicxx
CXXFLAGS  = -std=c++17 -O2 -Wall -fopenmp

# Executable name
TARGET    = laplace_solver

# Source and object files
SRCS      = main.cpp Grid2D.cpp LaplaceSolver.cpp
OBJS      = $(SRCS:.cpp=.o)

# Default run parameters (can be overridden on the command line)
NP      = 4
N       = 128
MAXITER = 20000
TOL     = 1e-6

# ---------------------------------------------------------------
# Default target: build the executable
# ---------------------------------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile each .cpp into a .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---------------------------------------------------------------
# Declare header dependencies so Make rebuilds when headers change
# ---------------------------------------------------------------
main.o:         main.cpp LaplaceSolver.hpp Grid2D.hpp
Grid2D.o:       Grid2D.cpp Grid2D.hpp
LaplaceSolver.o: LaplaceSolver.cpp LaplaceSolver.hpp Grid2D.hpp

# ---------------------------------------------------------------
# Run with default parameters (standard Jacobi)
# ---------------------------------------------------------------
run: all
	mpirun -np $(NP) ./$(TARGET) $(N) $(MAXITER) $(TOL)

# Run with Block Jacobi mode
run_block: all
	mpirun -np $(NP) ./$(TARGET) $(N) $(MAXITER) $(TOL) block

# ---------------------------------------------------------------
# Scalability test (calls the shell script in test/)
# ---------------------------------------------------------------
test: all
	bash test/run_scalability.sh

# ---------------------------------------------------------------
# Clean up compiled files
# ---------------------------------------------------------------
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run run_block test clean
