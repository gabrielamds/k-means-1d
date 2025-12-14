#!/bin/bash
# Script para gerar todos os datasets de teste

echo "Gerando datasets de teste..."

# Dataset pequeno: 10k pontos, 4 clusters
python3 generate_data.py --N 10000 --K 4 --output dados_pequeno --seed 42

# Dataset médio: 100k pontos, 8 clusters
python3 generate_data.py --N 100000 --K 8 --output dados_medio --seed 43

# Dataset grande: 1M pontos, 16 clusters
python3 generate_data.py --N 1000000 --K 16 --output dados_grande --seed 44

# Dataset extra grande: 10M pontos, 32 clusters (para testes de escalabilidade)
python3 generate_data.py --N 10000000 --K 32 --output dados_xlarge --seed 45

echo ""
echo "Todos os datasets gerados com sucesso!"
echo ""
echo "Uso:"
echo "  Serial:      ../serial/kmeans_1d_naive dados_pequeno.csv dados_pequeno_centroides_init.csv"
echo "  OpenMP:      ../openmp/kmeans_1d_parallel dados_medio.csv dados_medio_centroides_init.csv 50 1e-6 8"
echo "  CUDA:        ../cuda/kmeans_1d_cuda dados_grande.csv dados_grande_centroides_init.csv 50 1e-6 256"
echo "  MPI:         mpirun -np 4 ../mpi/kmeans_1d_mpi dados_grande.csv dados_grande_centroides_init.csv"
