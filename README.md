# Challenge 3 – Matrix-Free Parallel Solver for the Laplace Equation

## Overview

This project implements a **matrix-free parallel solver** for the 2D Laplace equation:

```
-Δu = 0   on  Ω = [0,1] × [0,1]
```

with user-defined Dirichlet boundary conditions, using the **Jacobi iterative method**.

Parallelisation is **hybrid MPI + OpenMP**:
- **MPI**: 1D row-wise domain decomposition with halo (ghost row) exchange.
- **OpenMP**: multi-threaded sweep inside each MPI process.

## Project Structure

```
challenge3/
├── CMakeLists.txt         # build configuration
├── main.cpp               # entry point, argument parsing, timing
├── Grid2D.hpp / .cpp      # 2D grid with ghost rows and BC support
├── LaplaceSolver.hpp/.cpp # Jacobi solver (standard + block Jacobi)
├── README.md
├── RESULT.md
└── test/
    └── run_scalability.sh # builds and runs scalability tests
```

## How to Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## How to Run

```bash
mpirun -np <nprocs> ./laplace_solver <n> <maxIter> <tol> [block]
```

| Argument  | Description                                      |
|-----------|--------------------------------------------------|
| `n`       | Global grid size (n × n, including boundaries)   |
| `maxIter` | Maximum number of Jacobi iterations              |
| `tol`     | Convergence tolerance (e.g. `1e-6`)              |
| `block`   | (Optional) Enable Block Jacobi / Schwarz mode    |

### Examples

```bash
# Standard Jacobi, 4 processes, 128×128 grid
mpirun -np 4 ./laplace_solver 128 5000 1e-6

# Block Jacobi (Schwarz), 4 processes
mpirun -np 4 ./laplace_solver 128 5000 1e-6 block
```

## Extras Implemented

### Extra 1 – User-defined Dirichlet BCs
The boundary condition is set via a `std::function<double(double x, double y)>` passed to the solver. The default function used is:

```
u(x, y) = sin(π x) · sinh(π y) / sinh(π)
```

which is the exact solution of the Laplace equation with u=0 on three sides and u=sin(πx) on the top side.

### Extra 2 – Block Jacobi / One-level Schwarz
When the `block` flag is passed, each MPI process performs `INNER_STEPS = 5` local Jacobi sweeps before exchanging halos. This naturally maps to a one-level Schwarz domain decomposition, trading slightly slower convergence for reduced communication overhead.

## Scalability Tests

```bash
bash test/run_scalability.sh
```

This script builds the project and runs it with 1, 2, and 4 MPI processes, printing elapsed time for both solver modes.
