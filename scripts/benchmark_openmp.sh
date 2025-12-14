#!/bin/bash
# Benchmark da versão OpenMP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OMP_BIN="$PROJECT_ROOT/openmp/kmeans_1d_parallel"
DATA_DIR="$PROJECT_ROOT/data"

MAX_ITER=50
EPS=1e-6

echo "=== Benchmark OpenMP ==="
echo ""

# Testa diferentes números de threads
THREADS=(1 2 4 8 16)
SCHEDULES=("static" "dynamic" "guided")

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
    
    # Varia número de threads com scheduling estático
    for T in "${THREADS[@]}"; do
        echo "  Threads: $T | Schedule: static"
        $OMP_BIN "$DATA_DIR/dados_$DATA.csv" "$DATA_DIR/dados_${DATA}_centroides_init.csv" \
                 $MAX_ITER $EPS $T static 0 2>&1 | grep -E "(Threads|Iterações|Tempo)"
        echo ""
    done
    
    echo "---"
    echo ""
done

# Teste de scheduling strategies com 8 threads no dataset médio
echo "Comparação de Scheduling Strategies (8 threads, dataset médio)"
echo ""
for SCHED in "${SCHEDULES[@]}"; do
    echo "  Schedule: $SCHED"
    $OMP_BIN "$DATA_DIR/dados_medio.csv" "$DATA_DIR/dados_medio_centroides_init.csv" \
             $MAX_ITER $EPS 8 $SCHED 0 2>&1 | grep -E "(Schedule|Tempo)"
    echo ""
done

echo "========================"
