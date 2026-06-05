# Challenge 3 – Matrix-Free Parallel Solver for the Laplace Equation

## Problem

We solve the Poisson problem on the unit square with Dirichlet boundary
conditions, using the **Jacobi iteration method**:

```
-Laplacian(u) = f(x,y)      in  (0,1) x (0,1)
            u = g(x,y)      on the boundary
```

The default test case (from the assignment) is

```
f(x,y) = 8*pi^2 * sin(2*pi*x) * sin(2*pi*y),     u = 0 on the boundary,
```

whose exact solution is `u(x,y) = sin(2*pi*x) * sin(2*pi*y)`. This lets us
measure the discrete **L2 error** against the known solution.

## Parallel design

- **MPI** – the global `n x n` grid is split into horizontal strips of rows
  (1D decomposition). The number of rows is balanced across the ranks. Each
  process exchanges its two border rows (ghost rows) with the neighbours using
  `MPI_Sendrecv`.
- **OpenMP** – inside each process the loop over the local rows of the Jacobi
  sweep is parallelised with `#pragma omp parallel for`.

## Files

```
challenge3/
├── Makefile               # build (mpicxx, -fopenmp, -std=c++17)
├── CMakeLists.txt         # alternative build (CLion / cmake)
├── main.cpp               # argument parsing, problem setup, timing, output
├── Grid2D.hpp / .cpp      # local grid with ghost rows and (i,j) indexing
├── LaplaceSolver.hpp/.cpp # Jacobi loops, MPI halo exchange, OpenMP, VTK, L2 error
├── .gitignore
├── README.md
└── test/
    ├── run_scalability.sh # builds and runs the full strong-scaling study
    ├── RESULT.md          # discussion of the results
    └── data/              # saved run logs, raw results and figures
        ├── dati_consegna.txt   # full cluster output (all grids, 1/2/4 procs)
        ├── figura.png          # time / speedup / efficiency plots
        └── fig3.fig            # MATLAB source of the plots
```

## Build

With the Makefile:

```bash
make
```

Or with CMake:

```bash
mkdir build && cd build
cmake .. && make
```

## Run

```bash
mpirun -np <nprocs> ./laplace_solver <n> <maxIter> <tol> [block] [bctest]
```

| Argument  | Meaning                                                   |
|-----------|-----------------------------------------------------------|
| `n`       | global grid size (n x n, boundaries included)             |
| `maxIter` | maximum number of Jacobi iterations                       |
| `tol`     | tolerance on the L2 residual                              |
| `block`   | (optional) use Block Jacobi / one-level Schwarz           |
| `bctest`  | (optional) non-homogeneous Dirichlet test (Extra 1)       |

Examples:

```bash
mpirun -np 4 ./laplace_solver 128 20000 1e-6           # standard Jacobi
mpirun -np 4 ./laplace_solver 128 20000 1e-6 block     # Block Jacobi (Schwarz)
mpirun -np 4 ./laplace_solver 128 20000 1e-6 bctest    # user-defined BCs
```

> On a cluster, set `OMP_NUM_THREADS=1` for a pure-MPI run and launch on a
> **compute node** (not the login node). Without `OMP_NUM_THREADS=1` each MPI
> process spawns one thread per core and oversubscribes the machine, which makes
> the run dramatically slower.

The program prints the number of iterations, the final residual, the L2 error
and the elapsed time. It also writes `solution.vtk`, which can be opened with
ParaView.

## Output

- `solution.vtk` – the solution on the whole grid (legacy VTK, structured
  points), gathered on rank 0 and written to file.

## Extras implemented

1. **User-defined Dirichlet boundary conditions** – the boundary values are
   given as a `std::function<double(double,double)>`. The `bctest` mode shows a
   non-homogeneous case (`g(x,y) = x + y`) with a known exact solution.
2. **Block Jacobi (one-level Schwarz)** – with the `block` flag, each process
   performs several local sweeps between two halo exchanges, i.e. it solves its
   own sub-domain more accurately before communicating.

## Scalability test

```bash
bash test/run_scalability.sh
```

This builds the code and runs the full strong-scaling study (every grid
`n = 16, 32, 64, 128, 256` with 1, 2 and 4 MPI processes), printing time,
speedup and efficiency. Run it on a **compute node**. The saved results are in
`test/data/` (`dati_consegna.txt`) and the plots in `test/data/figura.png`; the
discussion is in `test/RESULT.md`.
