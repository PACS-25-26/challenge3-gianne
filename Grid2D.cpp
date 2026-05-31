#include "Grid2D.hpp"
#include <algorithm>

Grid2D::Grid2D(int nx, int nLocal)
    : nx_(nx), nLocal_(nLocal),
      data_((nLocal + 2) * nx, 0.0)   // nLocal owned rows + 2 ghost rows
{}

double& Grid2D::at(int i, int j) {
    // i is in [0, nLocal+1], j is in [0, nx-1]
    return data_[i * nx_ + j];
}

const double& Grid2D::at(int i, int j) const {
    return data_[i * nx_ + j];
}

void Grid2D::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Grid2D::applyBoundaryConditions(
    std::function<double(double, double)> f,
    double xMin, double xMax,
    double yMin, double yMax,
    int globalRowStart,
    int nGlobal)
{
    // Physical step size (we have nGlobal interior points, so the mesh
    // spacing includes the two boundary endpoints: n+1 intervals -> n+2 points,
    // but here we treat nGlobal as the number of grid points including boundaries)
    double hx = (xMax - xMin) / (nGlobal - 1);
    double hy = (yMax - yMin) / (nx_   - 1);

    // Loop over owned rows (local rows 1 .. nLocal_, global rows globalRowStart .. globalRowStart+nLocal_-1)
    for (int i = 1; i <= nLocal_; ++i) {
        int globalRow = globalRowStart + (i - 1);
        double y = yMin + globalRow * hy;

        for (int j = 0; j < nx_; ++j) {
            double x = xMin + j * hx;

            // Left boundary (j == 0)
            if (j == 0)
                at(i, j) = f(x, y);

            // Right boundary (j == nx_-1)
            else if (j == nx_ - 1)
                at(i, j) = f(x, y);

            // Top boundary (globalRow == 0, i.e. first global row)
            else if (globalRow == 0)
                at(i, j) = f(x, y);

            // Bottom boundary (globalRow == nGlobal-1, i.e. last global row)
            else if (globalRow == nGlobal - 1)
                at(i, j) = f(x, y);
        }
    }
}
