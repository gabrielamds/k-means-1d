#!/bin/bash
# Benchmark da versão serial (baseline)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SERIAL_BIN="$PROJECT_ROOT/serial/kmeans_1d_naive"
DATA_DIR="$PROJECT_ROOT/data"

MAX_ITER=50
EPS=1e-6

echo "=== Benchmark Serial (Baseline) ==="
echo ""

# Dataset pequeno
echo "Dataset: PEQUENO (N=10k, K=4)"
$SERIAL_BIN "$DATA_DIR/dados_pequeno.csv" "$DATA_DIR/dados_pequeno_centroides_init.csv" $MAX_ITER $EPS

echo ""
echo "---"
echo ""

# Dataset médio
echo "Dataset: MÉDIO (N=100k, K=8)"
$SERIAL_BIN "$DATA_DIR/dados_medio.csv" "$DATA_DIR/dados_medio_centroides_init.csv" $MAX_ITER $EPS

echo ""
echo "---"
echo ""

# Dataset grande
echo "Dataset: GRANDE (N=1M, K=16)"
$SERIAL_BIN "$DATA_DIR/dados_grande.csv" "$DATA_DIR/dados_grande_centroides_init.csv" $MAX_ITER $EPS

echo ""
echo "==================================="
