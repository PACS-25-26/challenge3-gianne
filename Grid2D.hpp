#ifndef GRID2D_HPP
#define GRID2D_HPP

#include <vector>
#include <functional>
#include <stdexcept>

// Grid2D represents a 2D grid for a single MPI process.
// The global domain is an n x n grid. Each process owns a horizontal
// slice of rows, plus two "ghost" rows (top and bottom) for halo exchange.
//
//   ghost row (top)     <- row 0 in local storage
//   owned row 0         <- row 1
//   owned row 1         <- row 2
//   ...
//   owned row (nLocal-1)
//   ghost row (bottom)  <- last row in local storage
//
// So the local storage has (nLocal + 2) rows total.

class Grid2D {
public:
    // nx: number of columns (same as global n)
    // nLocal: number of rows owned by this process
    Grid2D(int nx, int nLocal);

    // Access element at local row i and column j (0-indexed, includes ghost rows)
    double& at(int i, int j);
    const double& at(int i, int j) const;

    // Fill the entire grid with a constant value (useful for initialization)
    void fill(double value);

    // Apply Dirichlet boundary conditions using a user-defined function.
    // The function f(x, y) returns the boundary value at physical coords (x, y).
    // xMin, xMax, yMin, yMax define the physical domain [0,1]x[0,1].
    // globalRowStart: the global row index of the first owned row of this process.
    // nGlobal: the total number of rows in the global grid.
    void applyBoundaryConditions(
        std::function<double(double, double)> f,
        double xMin, double xMax,
        double yMin, double yMax,
        int globalRowStart,
        int nGlobal
    );

    // Getters
    int getNx()     const { return nx_; }
    int getNLocal() const { return nLocal_; }

    // Raw access to underlying data (needed for MPI send/recv)
    double* rowPtr(int localRow) { return data_.data() + localRow * nx_; }
    const double* rowPtr(int localRow) const { return data_.data() + localRow * nx_; }

private:
    int nx_;       // number of columns
    int nLocal_;   // number of owned rows (excluding ghost rows)
    // Total rows stored = nLocal_ + 2 (one ghost on each side)
    std::vector<double> data_;
};

#endif // GRID2D_HPP
