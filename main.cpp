#include <iostream>
#include <string>
#include <cmath>
#include <mpi.h>

#include "LaplaceSolver.hpp"

// ------------------------------------------------------------------
// Usage:
//   mpirun -np <nprocs> ./laplace_solver <n> <maxIter> <tol> [block] [bctest]
//
//   n        : global grid size (n x n points, boundaries included)
//   maxIter  : maximum number of Jacobi iterations
//   tol      : convergence tolerance on the L2 residual (e.g. 1e-6)
//   block    : optional, enables Block Jacobi / one-level Schwarz
//   bctest   : optional, runs the non-homogeneous Dirichlet test (Extra 1)
//
// Examples:
//   mpirun -np 4 ./laplace_solver 128 20000 1e-6
//   mpirun -np 4 ./laplace_solver 128 20000 1e-6 block
//   mpirun -np 4 ./laplace_solver 128 20000 1e-6 bctest
// ------------------------------------------------------------------

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ---- Parse the command line ----
    if (argc < 4) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0]
                      << " <n> <maxIter> <tol> [block] [bctest]\n";
        MPI_Finalize();
        return 1;
    }

    int    n       = std::atoi(argv[1]);
    int    maxIter = std::atoi(argv[2]);
    double tol     = std::atof(argv[3]);

    // Look for the optional keywords among the remaining arguments.
    bool blockJacobi = false;
    bool bcTest      = false;
    for (int a = 4; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "block")  blockJacobi = true;
        if (arg == "bctest") bcTest      = true;
    }

    // ---- Choose the problem data ----
    // We use three functions: the forcing term f, the boundary values g,
    // and the known exact solution (to measure the error).
    LaplaceSolver::Function2D forcing, boundary, exact;

    if (!bcTest) {
        // Main problem from the assignment:
        //   -Laplacian(u) = 8*pi^2*sin(2*pi*x)*sin(2*pi*y),  u = 0 on the boundary.
        //   Exact solution: u(x,y) = sin(2*pi*x)*sin(2*pi*y).
        forcing  = [](double x, double y) {
            return 8.0 * M_PI * M_PI * std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y);
        };
        boundary = [](double, double) { return 0.0; };
        exact    = [](double x, double y) {
            return std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y);
        };
    } else {
        // Extra 1: user-defined NON-homogeneous Dirichlet boundary conditions.
        // We keep the SAME forcing term as the main problem, but now we impose
        // a simple, non-zero boundary function:  g(x,y) = x + y.
        //
        // We can still write down the exact solution. The function x + y is
        // "harmonic" (its Laplacian is zero), so by linearity the solution of
        //   -Laplacian(u) = f,   u = x + y on the boundary
        // is just the previous solution plus x + y:
        //   u(x,y) = sin(2*pi*x)*sin(2*pi*y) + (x + y).
        // This lets us measure the L2 error against a known exact solution.
        forcing  = [](double x, double y) {
            return 8.0 * M_PI * M_PI * std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y);
        };
        boundary = [](double x, double y) { return x + y; };
        exact    = [](double x, double y) {
            return std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y) + (x + y);
        };
    }

    // ---- Print a small header ----
    if (rank == 0) {
        std::cout << "=== Laplace Solver (Jacobi, MPI + OpenMP) ===\n";
        std::cout << "Grid       : " << n << " x " << n << "\n";
        std::cout << "Max iter   : " << maxIter << "\n";
        std::cout << "Tolerance  : " << tol << "\n";
        std::cout << "Mode       : " << (blockJacobi ? "Block Jacobi (Schwarz)" : "Standard Jacobi");
        if (bcTest) std::cout << "  [non-homogeneous BC test]";
        std::cout << "\n-------------------------------------------\n";
    }

    // ---- Build and run the solver ----
    LaplaceSolver solver(n, maxIter, tol, forcing, boundary, blockJacobi);

    double elapsed = solver.solve();              // timed part (MPI_Wtime)
    double error   = solver.computeL2Error(exact);
    solver.writeVTK("solution.vtk");

    // The slowest process determines the real wall-clock time.
    double maxTime = 0.0;
    MPI_Reduce(&elapsed, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Final report (rank 0) ----
    if (rank == 0) {
        std::cout << "-------------------------------------------\n";
        std::cout << "Iterations : " << solver.getIterations() << "\n";
        std::cout << "Residual   : " << solver.getResidual()   << "\n";
        std::cout << "L2 error   : " << error                  << "\n";
        std::cout << "Time (s)   : " << maxTime                << "\n";
    }

    MPI_Finalize();
    return 0;
}