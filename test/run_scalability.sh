#!/bin/bash
# ---------------------------------------------------------------
# run_scalability.sh
#
# Builds the code and runs the full strong-scaling study:
# every grid n in {16, 32, 64, 128, 256} with 1, 2 and 4 MPI processes.
# For each grid it prints the time, the speedup and the efficiency, and
# saves the full program logs under test/data/.
#
# Run it from anywhere:  bash test/run_scalability.sh
# IMPORTANT: run this on a COMPUTE node, not on the cluster login node.
# ---------------------------------------------------------------

set -e

# Move to the project root (the folder that contains the Makefile).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

DATA_DIR="test/data"
mkdir -p "$DATA_DIR"

echo "=== Building with make ==="
make
BIN=./laplace_solver

# One OpenMP thread per process, so "1 / 2 / 4 processes" really means
# "1 / 2 / 4 cores" (this measures the pure MPI scaling). Without this,
# each process would spawn one thread per core and oversubscribe the node.
export OMP_NUM_THREADS=1

MAXITER=20000
TOL=1e-6

# Small helper: extract a labelled value (e.g. "Time (s)") from a log file.
get_value() { grep "$1" "$2" | awk -F: '{print $2}' | tr -d ' '; }

# Full matrix: for every grid size, run with 1, 2 and 4 processes.
for N in 16 32 64 128 256; do
    echo ""
    echo "=== Grid $N x $N  (maxIter=$MAXITER, tol=$TOL) ==="
    printf "%-8s %-12s %-10s %-10s\n" "procs" "time[s]" "speedup" "efficiency"

    T1=""
    for NP in 1 2 4; do
        OUT="$DATA_DIR/run_n${N}_np${NP}.txt"
        mpirun -np $NP $BIN $N $MAXITER $TOL > "$OUT"

        T=$(get_value "Time (s)" "$OUT")
        if [ "$NP" -eq 1 ]; then T1=$T; fi

        # speedup = time(1)/time(p),  efficiency = speedup/p
        SP=$(awk -v a="$T1" -v b="$T"           'BEGIN{ if(b>0) printf "%.2f", a/b;     else print "-" }')
        EF=$(awk -v a="$T1" -v b="$T" -v p="$NP" 'BEGIN{ if(b>0) printf "%.2f", (a/b)/p; else print "-" }')

        printf "%-8s %-12s %-10s %-10s\n" "$NP" "$T" "$SP" "$EF"
    done
done

echo ""
echo "Done. Full logs are in $DATA_DIR/"
