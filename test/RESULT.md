# Results – Challenge 3

All the numbers below were obtained on the **cluster**, running on a compute
node (not the login node). The raw program output is saved in
`test/data/dati_consegna.txt`, and the plots of time / speedup / efficiency
are in `test/data/figura.png` (MATLAB source: `test/data/fig3.fig`).

---

## 1. Correctness check

The default problem is

```
-Laplacian(u) = 8*pi^2 sin(2*pi*x) sin(2*pi*y),    u = 0 on the boundary,
```

with exact solution `u(x,y) = sin(2*pi*x) sin(2*pi*y)`.

**Sanity check of the discretisation.** If we put the exact solution into the
grid and perform a single Jacobi sweep, the change (the local truncation error)
must go to zero as the grid is refined. This is exactly what we observe, which
confirms that the stencil and the `h^2 * f` scaling are correct:

| n   | h        | one-sweep residual |
|-----|----------|--------------------|
| 16  | 6.67e-2  | 1.26e-3            |
| 32  | 3.23e-2  | 7.00e-5            |
| 64  | 1.59e-2  | 4.12e-6            |
| 128 | 7.87e-3  | 2.50e-7            |
| 256 | 3.92e-3  | 2.00e-8            |

**Extra 1 (user-defined BCs).** Running with `bctest` keeps the same forcing
term but imposes the simple non-zero boundary data `g(x,y) = x + y`. Since
`x + y` is harmonic (its Laplacian is zero), by linearity the exact solution is
`u(x,y) = sin(2*pi*x) sin(2*pi*y) + (x + y)`. The measured L2 error is the same
as in the homogeneous case (about 9.5e-3 at n=32 and 3.4e-3 at n=64), which
confirms that the user-defined boundary conditions are handled correctly (the
linear part `x + y` is represented exactly by the 5-point stencil).

---

## 2. Accuracy vs grid size (`np` independent)

Standard Jacobi, `maxIter = 20000`, `tol = 1e-6`. These numbers are the same for
any number of processes (the algorithm is deterministic):

| n   | iterations | final residual | L2 error  |
|-----|------------|----------------|-----------|
| 16  | 134        | 9.3e-07        | 2.86e-02  |
| 32  | 530        | 9.9e-07        | 9.50e-03  |
| 64  | 1986       | 1.0e-06        | 3.09e-03  |
| 128 | 7220       | 1.0e-06        | 3.34e-04  |
| 256 | 20000 (max)| 5.6e-06        | 1.80e-02  |

**Discussion.**
- While Jacobi reaches the tolerance, the L2 error decreases with the grid size
  (from `n=16` to `n=128`), as expected for a second-order scheme.
- The number of iterations needed grows roughly like `n^2` (134 → 530 → 1986 →
  7220), which is the well-known slow convergence of Jacobi.
- At `n=256` the method does **not** reach the tolerance within 20000 iterations
  (residual still `5.6e-6`), so the solution is not fully converged and the L2
  error is large again. This matches the remark in the assignment: for large
  `n` one must allow many more iterations or use a better solver (multigrid or a
  preconditioned method).

---

## 3. Strong scalability (cluster)

Same problem solved with 1, 2 and 4 MPI processes (`OMP_NUM_THREADS=1`, so each
process uses one core). For each grid the three runs do exactly the same number
of iterations, so the comparison is fair. Reproduce with
`bash test/run_scalability.sh`; raw output in `test/data/dati_consegna.txt`.

**Execution time [s]**

| n   | p = 1    | p = 2   | p = 4   |
|-----|----------|---------|---------|
| 16  | 0.0064   | 0.0023  | 0.0143  |
| 32  | 0.0901   | 0.0753  | 0.1048  |
| 64  | 0.7725   | 0.4525  | 0.4568  |
| 128 | 12.019   | 8.240   | 6.777   |
| 256 | 133.94   | 78.42   | 50.46   |

**Speedup  S(p) = T(1) / T(p)**

| n   | p = 2 | p = 4 |
|-----|-------|-------|
| 16  | 2.75  | 0.45  |
| 32  | 1.20  | 0.86  |
| 64  | 1.71  | 1.69  |
| 128 | 1.46  | 1.77  |
| 256 | 1.71  | 2.65  |

**Efficiency  E(p) = S(p) / p**

| n   | p = 2  | p = 4  |
|-----|--------|--------|
| 16  | 1.38   | 0.11   |
| 32  | 0.60   | 0.21   |
| 64  | 0.85   | 0.42   |
| 128 | 0.73   | 0.44   |
| 256 | 0.85   | 0.66   |

![Execution time, speedup and efficiency vs grid size](data/figura.png)

*Figure: execution time (left), speedup (centre) and efficiency (right) versus
the grid size N, for 1, 2 and 4 cores. Speedup and efficiency are plotted for
the larger grids (N >= 64), where the timings are reliable; the dashed lines in
the centre plot are the ideal speedups (2 and 4).*

**Discussion.**
- The most important trend (clearly visible in the figure above): for a fixed number
  of processes, **the efficiency grows with the grid size**. The best result is
  at `n = 256`, where 4 processes give a speedup of **2.65** and an efficiency
  of **66%**.
- For **small grids** the parallel version is not worth it. At `n = 16` and
  `n = 32` each process gets only a handful of rows, so the cost of the halo
  exchange and of the global `MPI_Allreduce` (done every iteration) is larger
  than the actual computation: with 4 processes the program is even slower than
  the serial one. The `n = 16` timings (a few milliseconds) are also too small
  to be reliable, which is why `p = 2` there shows a meaningless "super-speedup".
- At `n = 64` going from 2 to 4 processes gives almost no gain (0.453 vs 0.457 s):
  the grid is still too small to keep 4 processes busy.
- The speedup is **sub-linear** (less than the ideal value of `p`) for two main
  reasons: the Jacobi stencil is **memory-bandwidth bound** (few operations, a
  lot of memory traffic), and every iteration contains a global `MPI_Allreduce`
  that synchronises all processes and does not get cheaper with more processes.
- Correctness is unaffected: the L2 error is identical for 1, 2 and 4 processes
  (Section 2), as expected for the standard Jacobi method.

### Reproducibility notes
- Always set `OMP_NUM_THREADS=1` for the pure-MPI scaling test. Without it each
  MPI process spawns one OpenMP thread per core, so `-np 4` creates 4 x (#cores)
  threads (oversubscription) and the run becomes dramatically slower.
- Run on a **compute node**, not on the login node. The login node is shared and
  gives only a couple of cores, so `-np 4` oversubscribes it and a single run can
  take minutes instead of seconds.

---

## 4. Block Jacobi vs standard Jacobi (Extra 2)

Dedicated back-to-back run, `n = 128`, `np = 4`, `maxIter = 20000`, `tol = 1e-6`:

| variant          | total sweeps | communications | L2 error | time [s] |
|------------------|--------------|----------------|----------|----------|
| Standard Jacobi  | 7220         | 7220           | 3.34e-4  | ~5.1     |
| Block Jacobi     | 8050         | 805            | 4.20e-4  | ~5.0     |

With `INNER_STEPS = 10`, Block Jacobi does 10 local sweeps per halo exchange, so
its 8050 sweeps correspond to only 8050 / 10 = 805 communication rounds. The
sweep and communication counts are exact (deterministic); the wall-times are
approximate and vary a little from run to run.

**Comment.**
- Block Jacobi needs slightly **more sweeps** (8050 vs 7220), because between two
  exchanges each process works with slightly out-of-date ghost rows.
- But it **communicates about 9 times less** (805 vs 7220 halo exchanges and
  `MPI_Allreduce` calls).
- Here the two effects almost cancel and the times are nearly equal. On a system
  where communication is more expensive (more nodes, slower network), the much
  smaller number of exchanges would make Block Jacobi the clear winner.
- Both variants solve the same problem; the small difference in L2 error just
  means they stopped at slightly different points.

Run and compare:

```bash
OMP_NUM_THREADS=1 mpirun -np 4 ./laplace_solver 128 20000 1e-6
OMP_NUM_THREADS=1 mpirun -np 4 ./laplace_solver 128 20000 1e-6 block
```
