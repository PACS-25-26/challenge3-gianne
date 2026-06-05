#include "LaplaceSolver.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------
// Helpers: how many rows and where they start for a given rank.
// We split n rows as evenly as possible: the first (n % size)
// ranks get one extra row.
// ---------------------------------------------------------
int LaplaceSolver::localRows(int rank) const {
    int base  = n_ / mpiSize_;
    int extra = n_ % mpiSize_;
    return base + (rank < extra ? 1 : 0);
}

int LaplaceSolver::rowStart(int rank) const {
    int base  = n_ / mpiSize_;
    int extra = n_ % mpiSize_;
    return rank * base + std::min(rank, extra);
}

// ---------------------------------------------------------
// Constructor: set up MPI topology and initialise the grids.
// ---------------------------------------------------------
LaplaceSolver::LaplaceSolver(int n, int maxIter, double tol,
                             Function2D forcing, Function2D boundary,
                             bool blockJacobi)
    : n_(n), maxIter_(maxIter), tol_(tol),
      blockJacobi_(blockJacobi),
      forcing_(forcing), boundary_(boundary),
      uOld_(0, 0), uNew_(0, 0)   // real size set a few lines below
{
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);

    // Mesh spacing on the unit square.
    h_ = 1.0 / (n_ - 1);

    // How many rows this process owns and where they start globally.
    nLocal_         = localRows(mpiRank_);
    globalRowStart_ = rowStart(mpiRank_);

    // Neighbours in the 1D chain of processes.
    rankTop_    = (mpiRank_ > 0)            ? mpiRank_ - 1 : MPI_PROC_NULL;
    rankBottom_ = (mpiRank_ < mpiSize_ - 1) ? mpiRank_ + 1 : MPI_PROC_NULL;

    // Build the two buffers with the correct local size.
    uOld_ = Grid2D(n_, nLocal_);
    uNew_ = Grid2D(n_, nLocal_);

    // Start from zero everywhere, then write the Dirichlet values.
    uOld_.fill(0.0);
    uNew_.fill(0.0);
    applyBoundaryConditions(uOld_);
    applyBoundaryConditions(uNew_);
}

// ---------------------------------------------------------
// applyBoundaryConditions: put g(x,y) on every boundary node.
// A node is on the boundary if it is on the first/last global row
// or on the first/last column.
// ---------------------------------------------------------
void LaplaceSolver::applyBoundaryConditions(Grid2D& g) {
    for (int i = 1; i <= nLocal_; ++i) {
        int globalRow = globalRowStart_ + (i - 1);
        double y = globalRow * h_;

        for (int j = 0; j < n_; ++j) {
            double x = j * h_;

            bool onBoundary = (globalRow == 0) || (globalRow == n_ - 1) ||
                              (j == 0)         || (j == n_ - 1);

            if (onBoundary)
                g.at(i, j) = boundary_(x, y);
        }
    }
}

// ---------------------------------------------------------
// haloExchange: send my border rows to the neighbours and receive
// theirs into my ghost rows. MPI_Sendrecv avoids deadlocks, and
// sending to MPI_PROC_NULL is simply ignored (handy at the domain ends).
// ---------------------------------------------------------
void LaplaceSolver::haloExchange() {
    // We use one tag per direction, so a row travelling "up" is never confused
    // with one travelling "down". This is what makes the send/receive pairs
    // match between neighbours (otherwise the processes deadlock).
    const int TAG_UP   = 0;   // a row sent towards the top neighbour
    const int TAG_DOWN = 1;   // a row sent towards the bottom neighbour

    // Exchange with the TOP neighbour:
    //   send my first owned row up (TAG_UP),
    //   receive into the top ghost row what the neighbour sent down (TAG_DOWN).
    MPI_Sendrecv(
        uOld_.rowPtr(1), n_, MPI_DOUBLE, rankTop_, TAG_UP,
        uOld_.rowPtr(0), n_, MPI_DOUBLE, rankTop_, TAG_DOWN,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Exchange with the BOTTOM neighbour:
    //   send my last owned row down (TAG_DOWN),
    //   receive into the bottom ghost row what the neighbour sent up (TAG_UP).
    MPI_Sendrecv(
        uOld_.rowPtr(nLocal_),     n_, MPI_DOUBLE, rankBottom_, TAG_DOWN,
        uOld_.rowPtr(nLocal_ + 1), n_, MPI_DOUBLE, rankBottom_, TAG_UP,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// ---------------------------------------------------------
// jacobiSweep: update every interior node as the average of its four
// neighbours plus the forcing term. Returns the local sum of squared
// increments, which feeds the L2 convergence test.
//
// Update formula (discretisation of -Laplacian(u) = f):
//   U_new(i,j) = 0.25 * ( U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1) + h^2 * f )
// ---------------------------------------------------------
double LaplaceSolver::jacobiSweep() {
    double localSumSq = 0.0;

    // OpenMP parallelises the loop over the rows owned by this process.
    // Each thread accumulates into localSumSq through a sum reduction.
#pragma omp parallel for reduction(+ : localSumSq) schedule(static)
    for (int i = 1; i <= nLocal_; ++i) {
        int globalRow = globalRowStart_ + (i - 1);

        // Skip the top and bottom boundary rows: they keep their BC values.
        if (globalRow == 0 || globalRow == n_ - 1)
            continue;

        double y = globalRow * h_;

        // Columns 0 and n-1 are boundary nodes, so we update only 1..n-2.
        for (int j = 1; j < n_ - 1; ++j) {
            double x = j * h_;

            double newVal = 0.25 * (
                uOld_.at(i - 1, j) +
                uOld_.at(i + 1, j) +
                uOld_.at(i, j - 1) +
                uOld_.at(i, j + 1) +
                h_ * h_ * forcing_(x, y)
            );

            uNew_.at(i, j) = newVal;

            double diff = newVal - uOld_.at(i, j);
            localSumSq += diff * diff;
        }
    }

    return localSumSq;
}

// ---------------------------------------------------------
// swapGrids: copy the new owned values back into the old buffer.
// (Boundary nodes are equal in both buffers, so copying them is harmless.)
// ---------------------------------------------------------
void LaplaceSolver::swapGrids() {
    for (int i = 1; i <= nLocal_; ++i)
        for (int j = 0; j < n_; ++j)
            uOld_.at(i, j) = uNew_.at(i, j);
}

// ---------------------------------------------------------
// solve: the main iteration loop. Returns the elapsed time.
// ---------------------------------------------------------
double LaplaceSolver::solve() {
    double startTime = MPI_Wtime();

    double e    = 0.0;   // global residual
    int    iter = 0;

    if (!blockJacobi_) {
        // ---- Standard point-wise Jacobi ----
        // One halo exchange + one sweep + one global check per iteration.
        for (iter = 0; iter < maxIter_; ++iter) {
            haloExchange();
            double localSumSq = jacobiSweep();
            swapGrids();

            // Global residual: sum the local contributions, then take sqrt(h*sum).
            double globalSumSq = 0.0;
            MPI_Allreduce(&localSumSq, &globalSumSq, 1, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
            e = std::sqrt(h_ * globalSumSq);

            if (mpiRank_ == 0 && iter % 200 == 0)
                std::cout << "iter " << iter << "   residual = " << e << "\n";

            if (e < tol_) break;
        }
    } else {
        // ---- Block Jacobi / one-level Schwarz ----
        // Each process does INNER_STEPS local sweeps between two exchanges.
        // We measure convergence on the FIRST sweep after the exchange:
        // when fresh neighbour data barely changes the solution, we are done.
        for (iter = 0; iter < maxIter_; iter += INNER_STEPS) {
            haloExchange();

            double firstSumSq = jacobiSweep();   // first local sweep
            swapGrids();

            for (int s = 1; s < INNER_STEPS; ++s) {   // extra local sweeps
                jacobiSweep();
                swapGrids();
            }

            double globalSumSq = 0.0;
            MPI_Allreduce(&firstSumSq, &globalSumSq, 1, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
            e = std::sqrt(h_ * globalSumSq);

            if (mpiRank_ == 0 && iter % 200 == 0)
                std::cout << "iter " << iter << "   residual = " << e << "  (block)\n";

            if (e < tol_) break;
        }
    }

    double endTime = MPI_Wtime();

    finalIter_     = iter;
    finalResidual_ = e;
    return endTime - startTime;
}

// ---------------------------------------------------------
// computeL2Error: discrete L2 distance from a known exact solution.
// ---------------------------------------------------------
double LaplaceSolver::computeL2Error(Function2D exact) const {
    double localSumSq = 0.0;

    for (int i = 1; i <= nLocal_; ++i) {
        int globalRow = globalRowStart_ + (i - 1);
        double y = globalRow * h_;

        for (int j = 0; j < n_; ++j) {
            double x = j * h_;
            double diff = uOld_.at(i, j) - exact(x, y);
            localSumSq += diff * diff;
        }
    }

    double globalSumSq = 0.0;
    MPI_Allreduce(&localSumSq, &globalSumSq, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    return std::sqrt(h_ * globalSumSq);
}

// ---------------------------------------------------------
// writeVTK: rank 0 gathers all the owned rows and writes a legacy VTK
// file (STRUCTURED_POINTS) that ParaView can open directly.
// ---------------------------------------------------------
void LaplaceSolver::writeVTK(const std::string& filename) const {
    // Pack my owned rows into a contiguous buffer.
    std::vector<double> sendBuf(nLocal_ * n_);
    for (int i = 1; i <= nLocal_; ++i)
        for (int j = 0; j < n_; ++j)
            sendBuf[(i - 1) * n_ + j] = uOld_.at(i, j);

    // Rank 0 prepares the receive counts and offsets for MPI_Gatherv.
    std::vector<int>    recvCounts;
    std::vector<int>    displs;
    std::vector<double> full;

    if (mpiRank_ == 0) {
        recvCounts.resize(mpiSize_);
        displs.resize(mpiSize_);
        int offset = 0;
        for (int r = 0; r < mpiSize_; ++r) {
            recvCounts[r] = localRows(r) * n_;
            displs[r]     = offset;
            offset       += recvCounts[r];
        }
        full.resize(n_ * n_);
    }

    MPI_Gatherv(sendBuf.data(), nLocal_ * n_, MPI_DOUBLE,
                full.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Only rank 0 writes the file.
    if (mpiRank_ == 0) {
        std::ofstream out(filename);
        out << "# vtk DataFile Version 3.0\n";
        out << "Laplace equation solution\n";
        out << "ASCII\n";
        out << "DATASET STRUCTURED_POINTS\n";
        out << "DIMENSIONS " << n_ << " " << n_ << " 1\n";
        out << "ORIGIN 0 0 0\n";
        out << "SPACING " << h_ << " " << h_ << " 1\n";
        out << "POINT_DATA " << n_ * n_ << "\n";
        out << "SCALARS solution double 1\n";
        out << "LOOKUP_TABLE default\n";

        // VTK expects the x index to vary fastest, then y.
        for (int row = 0; row < n_; ++row)
            for (int col = 0; col < n_; ++col)
                out << full[row * n_ + col] << "\n";

        out.close();
        std::cout << "Solution written to " << filename << "\n";
    }
}
