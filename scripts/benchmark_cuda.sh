#!/bin/bash
# Benchmark da versão CUDA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CUDA_BIN="$PROJECT_ROOT/cuda/kmeans_1d_cuda"
DATA_DIR="$PROJECT_ROOT/data"

MAX_ITER=50
EPS=1e-6

echo "=== Benchmark CUDA ==="
echo ""

# Testa diferentes block sizes
BLOCK_SIZES=(128 256 512)

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
    
    for BS in "${BLOCK_SIZES[@]}"; do
        echo "  Block size: $BS"
        $CUDA_BIN "$DATA_DIR/dados_$DATA.csv" "$DATA_DIR/dados_${DATA}_centroides_init.csv" \
                  $MAX_ITER $EPS $BS 2>&1 | grep -E "(Block size|Iterações|H2D|Kernel|D2H|Tempo total|Throughput)"
        echo ""
    done
    
    echo "---"
    echo ""
done

echo "======================"
