#include "LaplaceSolver.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------
// Constructor: set up MPI topology and initialize grids
// ---------------------------------------------------------
LaplaceSolver::LaplaceSolver(int n, int maxIter, double tol,
                             std::function<double(double, double)> bcFunc,
                             bool blockJacobi)
    : n_(n), maxIter_(maxIter), tol_(tol),
      bcFunc_(bcFunc), blockJacobi_(blockJacobi),
      uOld_(0, 0), uNew_(0, 0)   // will be rebuilt below
{
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);

    // Distribute rows as evenly as possible across processes.
    // The first (n % mpiSize_) processes get one extra row.
    int base  = n_ / mpiSize_;
    int extra = n_ % mpiSize_;

    nLocal_         = base + (mpiRank_ < extra ? 1 : 0);
    globalRowStart_ = mpiRank_ * base + std::min(mpiRank_, extra);

    // Neighbours in the 1D chain
    rankTop_    = (mpiRank_ > 0)            ? mpiRank_ - 1 : MPI_PROC_NULL;
    rankBottom_ = (mpiRank_ < mpiSize_ - 1) ? mpiRank_ + 1 : MPI_PROC_NULL;

    // Build grids with the correct local size
    uOld_ = Grid2D(n_, nLocal_);
    uNew_ = Grid2D(n_, nLocal_);

    // Initialize interior to zero; boundaries will be set by applyBoundaryConditions
    uOld_.fill(0.0);
    uNew_.fill(0.0);

    // Apply user-defined Dirichlet BCs to both buffers
    uOld_.applyBoundaryConditions(bcFunc_, 0.0, 1.0, 0.0, 1.0, globalRowStart_, n_);
    uNew_.applyBoundaryConditions(bcFunc_, 0.0, 1.0, 0.0, 1.0, globalRowStart_, n_);
}

// ---------------------------------------------------------
// haloExchange: send/receive ghost rows with MPI neighbours
// ---------------------------------------------------------
void LaplaceSolver::haloExchange() {
    // We use MPI_Sendrecv to avoid deadlocks.
    // This process sends its first owned row (local row 1) to rankTop_,
    // and receives from rankTop_ into ghost row 0.
    // Similarly for the bottom.

    MPI_Sendrecv(
        uOld_.rowPtr(1),           // send: my first owned row
        n_, MPI_DOUBLE, rankTop_, 0,
        uOld_.rowPtr(0),           // recv: ghost row at top
        n_, MPI_DOUBLE, rankTop_, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        uOld_.rowPtr(nLocal_),     // send: my last owned row
        n_, MPI_DOUBLE, rankBottom_, 1,
        uOld_.rowPtr(nLocal_ + 1), // recv: ghost row at bottom
        n_, MPI_DOUBLE, rankBottom_, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

// ---------------------------------------------------------
// jacobiSweep: one sweep of the Jacobi update on interior nodes
// Returns the local maximum change (residual).
// ---------------------------------------------------------
double LaplaceSolver::jacobiSweep() {
    double localRes = 0.0;

    // Loop over owned rows (local indices 1 .. nLocal_).
    // Row i corresponds to global row (globalRowStart_ + i - 1).
    // Parallelise with OpenMP across the row loop.

#pragma omp parallel for reduction(max : localRes) schedule(static)
    for (int i = 1; i <= nLocal_; ++i) {
        int globalRow = globalRowStart_ + (i - 1);

        for (int j = 1; j < n_ - 1; ++j) {
            // Skip boundary nodes (left/right handled outside, top/bottom by BCs)
            if (globalRow == 0 || globalRow == n_ - 1)
                continue;

            // Standard 5-point Laplacian stencil (matrix-free):
            // u_new[i,j] = 0.25 * (u_old[i-1,j] + u_old[i+1,j] + u_old[i,j-1] + u_old[i,j+1])
            double newVal = 0.25 * (
                uOld_.at(i - 1, j) +
                uOld_.at(i + 1, j) +
                uOld_.at(i, j - 1) +
                uOld_.at(i, j + 1)
            );

            uNew_.at(i, j) = newVal;

            // Track the maximum change for convergence
            double diff = std::abs(newVal - uOld_.at(i, j));
            if (diff > localRes)
                localRes = diff;
        }
    }

    return localRes;
}

// ---------------------------------------------------------
// swapGrids: copy uNew into uOld for the next iteration
// ---------------------------------------------------------
void LaplaceSolver::swapGrids() {
    // Only copy interior owned rows, not ghost rows (they will be refreshed by haloExchange)
    for (int i = 1; i <= nLocal_; ++i)
        for (int j = 0; j < n_; ++j)
            uOld_.at(i, j) = uNew_.at(i, j);
}

// ---------------------------------------------------------
// globalResidual: MPI_Allreduce to get the maximum residual across all processes
// ---------------------------------------------------------
double LaplaceSolver::globalResidual(double localRes) const {
    double globalRes = 0.0;
    MPI_Allreduce(&localRes, &globalRes, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return globalRes;
}

// ---------------------------------------------------------
// solve: main iteration loop
// ---------------------------------------------------------
double LaplaceSolver::solve() {
    double startTime = MPI_Wtime();

    if (!blockJacobi_) {
        // ---- Standard point-wise Jacobi ----
        for (int iter = 0; iter < maxIter_; ++iter) {
            // 1. Exchange ghost rows so each process has up-to-date neighbours
            haloExchange();

            // 2. Perform the Jacobi sweep
            double localRes = jacobiSweep();

            // 3. Copy uNew -> uOld for next iteration
            swapGrids();

            // 4. Check global convergence (expensive: one MPI_Allreduce per iteration)
            double res = globalResidual(localRes);

            if (mpiRank_ == 0 && iter % 100 == 0)
                std::cout << "Iter " << iter << "  residual = " << res << "\n";

            if (res < tol_) {
                if (mpiRank_ == 0)
                    std::cout << "Converged at iteration " << iter
                              << "  residual = " << res << "\n";
                break;
            }
        }
    } else {
        // ---- Block Jacobi / one-level Schwarz ----
        // Each MPI process iterates locally for INNER_STEPS steps
        // before performing a halo exchange. This reduces communication
        // frequency but slows convergence slightly.
        for (int iter = 0; iter < maxIter_; iter += INNER_STEPS) {

            // Exchange once per INNER_STEPS outer iterations
            haloExchange();

            double localRes = 0.0;
            for (int inner = 0; inner < INNER_STEPS; ++inner) {
                localRes = jacobiSweep();
                swapGrids();
            }

            double res = globalResidual(localRes);

            if (mpiRank_ == 0 && iter % 100 == 0)
                std::cout << "Iter " << iter << "  residual = " << res << " (Block Jacobi)\n";

            if (res < tol_) {
                if (mpiRank_ == 0)
                    std::cout << "Converged at iteration " << iter
                              << "  residual = " << res << " (Block Jacobi)\n";
                break;
            }
        }
    }

    double endTime = MPI_Wtime();
    return endTime - startTime;
}

// ---------------------------------------------------------
// printSolution: print the grid owned by each process in order (rank 0 first)
// ---------------------------------------------------------
void LaplaceSolver::printSolution() const {
    for (int rank = 0; rank < mpiSize_; ++rank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpiRank_ == rank) {
            for (int i = 1; i <= nLocal_; ++i) {
                for (int j = 0; j < n_; ++j)
                    std::cout << uOld_.at(i, j) << " ";
                std::cout << "\n";
            }
        }
    }
}
