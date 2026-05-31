#ifndef LAPLACESOLVER_HPP
#define LAPLACESOLVER_HPP

#include "Grid2D.hpp"
#include <functional>
#include <mpi.h>

// LaplaceSolver solves the 2D Laplace equation  -Δu = 0
// on the unit square [0,1]x[0,1] with Dirichlet boundary conditions,
// using the Jacobi iterative method.
//
// Parallelization:
//   - MPI: 1D row-wise domain decomposition. Each process owns a strip of rows.
//     Halo exchange (ghost rows) is done with MPI_Sendrecv between neighbours.
//   - OpenMP: the inner Jacobi sweep is parallelised over rows using
//     #pragma omp parallel for.
//
// Two modes are available (selected by the blockJacobi flag):
//   - Standard Jacobi: classic point-wise update. Convergence check is global.
//   - Block Jacobi (one-level Schwarz): each MPI process iterates locally
//     for several inner steps before exchanging halos. This reduces communication
//     frequency and mimics the one-level Schwarz decomposition.

class LaplaceSolver {
public:
    // n         : global grid size (n x n points, including boundaries)
    // maxIter   : maximum number of Jacobi iterations
    // tol       : convergence tolerance (||u_new - u_old||_inf < tol)
    // bcFunc    : user-defined Dirichlet BC function f(x, y)
    // blockJacobi : if true, use Block Jacobi / one-level Schwarz variant
    LaplaceSolver(int n, int maxIter, double tol,
                  std::function<double(double, double)> bcFunc,
                  bool blockJacobi = false);

    // Run the solver. Returns the elapsed wall-clock time (MPI_Wtime).
    double solve();

    // Write the solution of the root process to stdout (for small grids / debugging)
    void printSolution() const;

private:
    // MPI topology information
    int mpiRank_;        // rank of this process
    int mpiSize_;        // total number of processes
    int rankTop_;        // rank of the neighbour above  (-1 if none)
    int rankBottom_;     // rank of the neighbour below  (-1 if none)

    // Grid parameters
    int n_;              // global grid size (n x n)
    int nLocal_;         // number of rows owned by this process
    int globalRowStart_; // first global row index owned by this process

    // Solver parameters
    int    maxIter_;
    double tol_;
    bool   blockJacobi_;

    // Number of inner iterations per halo exchange in Block Jacobi mode
    static constexpr int INNER_STEPS = 5;

    // Boundary condition function
    std::function<double(double, double)> bcFunc_;

    // The two grid buffers (old and new solution)
    Grid2D uOld_;
    Grid2D uNew_;

    // --- Private helpers ---

    // Perform one full Jacobi sweep on the local slice.
    // Returns the local maximum change ||u_new - u_old||_inf on interior nodes.
    double jacobiSweep();

    // Exchange ghost (halo) rows with MPI neighbours.
    void haloExchange();

    // Copy u_new back into u_old (swap contents).
    void swapGrids();

    // Global convergence check: reduce local residual across all processes.
    double globalResidual(double localRes) const;
};

#endif // LAPLACESOLVER_HPP
