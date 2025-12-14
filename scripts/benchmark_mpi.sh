#!/bin/bash
# Benchmark da versão MPI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MPI_BIN="$PROJECT_ROOT/mpi/kmeans_1d_mpi"
DATA_DIR="$PROJECT_ROOT/data"

MAX_ITER=50
EPS=1e-6

echo "=== Benchmark MPI ==="
echo ""

# Testa diferentes números de processos
PROCS=(1 2 4 8)

for DATA in "pequeno" "medio" "grande"; do
    if [ "$DATA" == "pequeno" ]; then
        N="10k"
        K=4
    elif [ "$DATA" == "medio" ]; then
        N="100k"
        K=8
    else
        N="1M"
        K=16
    fi
    
    echo "Dataset: ${DATA^^} (N=$N, K=$K)"
    echo ""
    
    for P in "${PROCS[@]}"; do
        echo "  Processos MPI: $P"
        mpirun -np $P $MPI_BIN "$DATA_DIR/dados_$DATA.csv" "$DATA_DIR/dados_${DATA}_centroides_init.csv" \
                               $MAX_ITER $EPS 2>&1 | grep -E "(Processos|Iterações|Tempo|comunicação)"
        echo ""
    done
    
    echo "---"
    echo ""
done

echo "====================="
