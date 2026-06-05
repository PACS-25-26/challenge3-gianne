#include "Grid2D.hpp"
#include <algorithm>

Grid2D::Grid2D(int nx, int nLocal)
    : nx_(nx), nLocal_(nLocal),
      data_((nLocal + 2) * nx, 0.0)   // nLocal owned rows + 2 ghost rows
{}

double& Grid2D::at(int i, int j) {
    // i goes from 0 to nLocal_+1 (including the two ghost rows)
    // j goes from 0 to nx_-1
    return data_[i * nx_ + j];
}

const double& Grid2D::at(int i, int j) const {
    return data_[i * nx_ + j];
}

void Grid2D::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}
