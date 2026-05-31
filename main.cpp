#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

#include "LaplaceSolver.hpp"

// ------------------------------------------------------------------
// Usage:
//   mpirun -np <nprocs> ./laplace_solver <n> <maxIter> <tol> [block]
//
//   n        : global grid size (n x n), including boundary points
//   maxIter  : maximum number of Jacobi iterations
//   tol      : convergence tolerance (e.g. 1e-6)
//   block    : optional string "block" to enable Block Jacobi / Schwarz
//
// Example:
//   mpirun -np 4 ./laplace_solver 128 5000 1e-6 block
// ------------------------------------------------------------------

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ---- Parse command-line arguments ----
    if (argc < 4) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0]
                      << " <n> <maxIter> <tol> [block]\n";
        MPI_Finalize();
        return 1;
    }

    int    n          = std::atoi(argv[1]);
    int    maxIter    = std::atoi(argv[2]);
    double tol        = std::atof(argv[3]);
    bool   blockJacobi = (argc >= 5 && std::string(argv[4]) == "block");

    if (rank == 0) {
        std::cout << "=== Laplace Solver (Jacobi, MPI+OpenMP) ===\n";
        std::cout << "Grid size   : " << n << " x " << n << "\n";
        std::cout << "Max iter    : " << maxIter << "\n";
        std::cout << "Tolerance   : " << tol << "\n";
        std::cout << "Mode        : " << (blockJacobi ? "Block Jacobi (Schwarz)" : "Standard Jacobi") << "\n";
        std::cout << "-------------------------------------------\n";
    }

    // ---- Define Dirichlet boundary conditions (Extra 1) ----
    // Here we use a simple harmonic function u(x,y) = sin(pi*x)*sinh(pi*y) / sinh(pi)
    // as an example. Any std::function<double(double,double)> works.
    auto bcFunc = [](double x, double y) -> double {
        // This exact solution satisfies -Delta u = 0
        // with u = 0 on three sides and u = sin(pi*x) on the top side (y=1)
        return std::sin(M_PI * x) * std::sinh(M_PI * y) / std::sinh(M_PI);
    };

    // ---- Run the solver ----
    LaplaceSolver solver(n, maxIter, tol, bcFunc, blockJacobi);

    double elapsed = solver.solve();   // returns MPI_Wtime elapsed time

    // ---- Report timing (rank 0 collects and prints) ----
    // We report the maximum time across all processes (the bottleneck)
    double maxTime = 0.0;
    MPI_Reduce(&elapsed, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "-------------------------------------------\n";
        std::cout << "Elapsed time (computation + comm): "
                  << maxTime << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
