#ifndef GRID2D_HPP
#define GRID2D_HPP

#include <vector>

// Grid2D stores the part of the global grid owned by one MPI process.
//
// The global problem is an n x n grid of points. With the 1D row-wise
// decomposition, each process owns a horizontal strip of "nLocal" rows.
// We also keep two extra "ghost" rows (one on top, one on the bottom)
// that hold a copy of the neighbour rows received via MPI. This way the
// 5-point stencil can always read the row above and below.
//
// Local row layout (local index -> meaning):
//   row 0            -> ghost row (copy of the neighbour above)
//   row 1            -> first owned row
//   ...
//   row nLocal       -> last owned row
//   row nLocal + 1   -> ghost row (copy of the neighbour below)
//
// So the storage has (nLocal + 2) rows, each with nx columns.
// Data is kept in a single flat std::vector (matrix-free, no sparse matrix).

class Grid2D {
public:
    Grid2D(int nx, int nLocal);

    // Read/write the value at local row i and column j.
    double& at(int i, int j);
    const double& at(int i, int j) const;

    // Set every entry to the same value.
    void fill(double value);

    // Getters
    int getNx()     const { return nx_; }
    int getNLocal() const { return nLocal_; }

    // Pointer to the first element of a given local row.
    // Needed to pass a whole row to MPI send/receive.
    double*       rowPtr(int localRow)       { return data_.data() + localRow * nx_; }
    const double* rowPtr(int localRow) const { return data_.data() + localRow * nx_; }

private:
    int nx_;       // number of columns (same as the global n)
    int nLocal_;   // number of owned rows (without the 2 ghost rows)
    std::vector<double> data_;   // size = (nLocal_ + 2) * nx_
};

#endif // GRID2D_HPP
