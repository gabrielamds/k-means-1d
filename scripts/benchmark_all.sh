#!/bin/bash
# Script mestre para rodar todos os benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   K-means 1D - Benchmark Completo     ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Cria diretório de resultados
mkdir -p "$RESULTS_DIR"

# Verifica se os dados existem
if [ ! -f "$PROJECT_ROOT/data/dados_pequeno.csv" ]; then
    echo -e "${YELLOW}Dados não encontrados. Gerando...${NC}"
    cd "$PROJECT_ROOT/data"
    bash generate_datasets.sh
    cd "$PROJECT_ROOT"
fi

# Parâmetros comuns
MAX_ITER=50
EPS=1e-6

echo -e "${GREEN}[1/5] Benchmarking Serial (baseline)...${NC}"
bash "$SCRIPT_DIR/benchmark_serial.sh" | tee "$RESULTS_DIR/serial_results.txt"

echo ""
echo -e "${GREEN}[2/5] Benchmarking OpenMP...${NC}"
bash "$SCRIPT_DIR/benchmark_openmp.sh" | tee "$RESULTS_DIR/openmp_results.txt"

echo ""
echo -e "${GREEN}[3/5] Benchmarking CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    bash "$SCRIPT_DIR/benchmark_cuda.sh" | tee "$RESULTS_DIR/cuda_results.txt"
else
    echo -e "${YELLOW}CUDA não disponível, pulando...${NC}"
    echo "CUDA not available" > "$RESULTS_DIR/cuda_results.txt"
fi

echo ""
echo -e "${GREEN}[4/5] Benchmarking MPI...${NC}"
if command -v mpirun &> /dev/null; then
    bash "$SCRIPT_DIR/benchmark_mpi.sh" | tee "$RESULTS_DIR/mpi_results.txt"
else
    echo -e "${YELLOW}MPI não disponível, pulando...${NC}"
    echo "MPI not available" > "$RESULTS_DIR/mpi_results.txt"
fi

echo ""
echo -e "${GREEN}[5/5] Benchmarking Híbridos...${NC}"
bash "$SCRIPT_DIR/benchmark_hybrid.sh" | tee "$RESULTS_DIR/hybrid_results.txt"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}   Benchmarks Completos!               ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Resultados salvos em: $RESULTS_DIR"
echo ""
echo "Próximos passos:"
echo "  1. python3 analysis/analyze_speedup.py"
echo "  2. python3 analysis/generate_report.py"
