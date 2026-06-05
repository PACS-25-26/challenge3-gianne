#ifndef LAPLACESOLVER_HPP
#define LAPLACESOLVER_HPP

#include "Grid2D.hpp"
#include <functional>
#include <string>
#include <mpi.h>

// LaplaceSolver solves the Poisson/Laplace problem
//
//     -Laplacian(u) = f(x,y)   inside the unit square (0,1)x(0,1)
//                 u = g(x,y)   on the boundary  (Dirichlet)
//
// with the Jacobi iterative method.
//
// Parallel design:
//   - MPI: the global grid is split into horizontal strips of rows.
//          Each process works on its own strip and exchanges the two
//          border rows (halo / ghost rows) with its neighbours.
//   - OpenMP: the loop over the local rows in the Jacobi sweep is
//          parallelised with "#pragma omp parallel for".
//
// Two iteration modes (chosen with the blockJacobi flag):
//   - Standard Jacobi: one sweep, then one halo exchange, every iteration.
//   - Block Jacobi (one-level Schwarz): each process does several local
//          sweeps before exchanging halos. The local sub-domain is solved
//          more accurately between two communications.

class LaplaceSolver {
public:
    // A function of two variables, used for f, the boundary values and the
    // exact solution.
    using Function2D = std::function<double(double, double)>;

    LaplaceSolver(int n, int maxIter, double tol,
                  Function2D forcing, Function2D boundary,
                  bool blockJacobi = false);

    // Run the Jacobi iterations. Returns the elapsed wall-clock time
    // (measured with MPI_Wtime, only around the computation).
    double solve();

    // Compute the discrete L2 error between the numerical solution and a
    // known exact solution
    double computeL2Error(Function2D exact) const;

    // Collect the full solution on rank 0 and write it to a VTK file
    // that can be opened with ParaView.
    void writeVTK(const std::string& filename) const;

    // Information about the last run.
    int    getIterations() const { return finalIter_; }
    double getResidual()   const { return finalResidual_; }

private:
    // --- MPI topology ---
    int mpiRank_;      // this process rank
    int mpiSize_;      // total number of processes
    int rankTop_;      // neighbour above  (MPI_PROC_NULL if none)
    int rankBottom_;   // neighbour below  (MPI_PROC_NULL if none)

    // --- Grid description ---
    int    n_;               // global grid size (n x n)
    int    nLocal_;          // number of rows owned by this process
    int    globalRowStart_;  // global index of this process' first owned row
    double h_;               // mesh spacing, h = 1/(n-1)

    // --- Solver parameters ---
    int    maxIter_;
    double tol_;
    bool   blockJacobi_;

    // Number of local sweeps per halo exchange in Block Jacobi mode.
    static constexpr int INNER_STEPS = 10;

    // --- Problem data (passed by the user) ---
    Function2D forcing_;    // right-hand side f(x,y)
    Function2D boundary_;   // Dirichlet boundary values g(x,y)

    // --- The two solution buffers ---
    Grid2D uOld_;
    Grid2D uNew_;

    // --- Results of the last run ---
    int    finalIter_     = 0;
    double finalResidual_ = 0.0;

    // --- Helpers ---
    // Write the Dirichlet values on the boundary nodes of a grid.
    void applyBoundaryConditions(Grid2D& g);

    // One Jacobi sweep over the owned interior nodes.
    // Returns the local sum of squared increments (used for the L2 residual).
    double jacobiSweep();

    // Exchange the border rows with the top and bottom neighbours.
    void haloExchange();

    // Copy the new solution into the old buffer for the next iteration.
    void swapGrids();

    // Number of rows and starting row owned by a given rank
    // (same balanced splitting used in the constructor).
    int localRows(int rank) const;
    int rowStart(int rank) const;
};

#endif // LAPLACESOLVER_HPP

