#!/bin/bash
# Benchmark das versões híbridas

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_DIR="$PROJECT_ROOT/data"
MAX_ITER=50
EPS=1e-6

echo "=== Benchmark Híbridos ==="
echo ""

# OpenMP + CUDA
if [ -f "$PROJECT_ROOT/hybrid/kmeans_1d_omp_cuda" ]; then
    echo "[1] OpenMP + CUDA"
    echo ""
    
    for THREADS in 2 4; do
        for BS in 256; do
            echo "  Threads: $THREADS | Block size: $BS | Dataset: MÉDIO"
            $PROJECT_ROOT/hybrid/kmeans_1d_omp_cuda "$DATA_DIR/dados_medio.csv" \
                "$DATA_DIR/dados_medio_centroides_init.csv" $MAX_ITER $EPS $THREADS $BS 2>&1 | \
                grep -E "(Threads|Block size|Iterações|Tempo)"
            echo ""
        done
    done
    echo "---"
    echo ""
fi

# OpenMP + MPI
if [ -f "$PROJECT_ROOT/hybrid/kmeans_1d_omp_mpi" ]; then
    echo "[2] OpenMP + MPI"
    echo ""
    
    for PROCS in 2 4; do
        for THREADS in 2 4; do
            echo "  Processos: $PROCS | Threads: $THREADS | Total: $((PROCS*THREADS)) | Dataset: MÉDIO"
            mpirun -np $PROCS $PROJECT_ROOT/hybrid/kmeans_1d_omp_mpi "$DATA_DIR/dados_medio.csv" \
                "$DATA_DIR/dados_medio_centroides_init.csv" $MAX_ITER $EPS $THREADS 2>&1 | \
                grep -E "(Processos|Threads|Total workers|Iterações|Tempo)"
            echo ""
        done
    done
    echo "---"
    echo ""
fi

# MPI + CUDA
if [ -f "$PROJECT_ROOT/hybrid/kmeans_1d_mpi_cuda" ]; then
    echo "[3] MPI + CUDA"
    echo ""
    
    for PROCS in 2 4; do
        for BS in 256; do
            echo "  Processos: $PROCS | Block size: $BS | Dataset: GRANDE"
            mpirun -np $PROCS $PROJECT_ROOT/hybrid/kmeans_1d_mpi_cuda "$DATA_DIR/dados_grande.csv" \
                "$DATA_DIR/dados_grande_centroides_init.csv" $MAX_ITER $EPS $BS 2>&1 | \
                grep -E "(Processos|GPUs|Block size|Iterações|Tempo)"
            echo ""
        done
    done
    echo "---"
    echo ""
fi

echo "==========================="
