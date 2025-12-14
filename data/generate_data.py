#!/usr/bin/env python3
"""
Script para gerar dados sintéticos para testes de K-means 1D
Gera N pontos distribuídos em K clusters gaussianos
"""

import numpy as np
import argparse
import sys

def generate_kmeans_data(N, K, seed=42):
    """
    Gera N pontos 1D distribuídos em K clusters
    
    Args:
        N: número de pontos
        K: número de clusters
        seed: seed para reprodutibilidade
    
    Returns:
        data: array de N pontos
        true_centroids: centróides verdadeiros dos clusters
    """
    np.random.seed(seed)
    
    # Gera centróides verdadeiros espaçados
    true_centroids = np.linspace(0, 100, K)
    
    # Adiciona ruído aos centróides
    true_centroids += np.random.randn(K) * 5
    
    # Gera pontos para cada cluster
    points_per_cluster = N // K
    remainder = N % K
    
    data = []
    for i in range(K):
        n_points = points_per_cluster + (1 if i < remainder else 0)
        # Gera pontos gaussianos ao redor do centróide
        cluster_points = np.random.randn(n_points) * 3 + true_centroids[i]
        data.extend(cluster_points)
    
    data = np.array(data)
    np.random.shuffle(data)  # Embaralha para não estar ordenado por cluster
    
    return data, true_centroids

def generate_initial_centroids(K, data_range, seed=42):
    """
    Gera K centróides iniciais aleatórios
    
    Args:
        K: número de centróides
        data_range: tupla (min, max) do range dos dados
        seed: seed para reprodutibilidade
    
    Returns:
        centroids: array de K centróides
    """
    np.random.seed(seed + 1)
    return np.random.uniform(data_range[0], data_range[1], K)

def save_csv_1col(filename, data):
    """Salva dados em formato CSV com 1 coluna (sem cabeçalho)"""
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')

def main():
    parser = argparse.ArgumentParser(description='Gera dados sintéticos para K-means 1D')
    parser.add_argument('--N', type=int, required=True, help='Número de pontos')
    parser.add_argument('--K', type=int, required=True, help='Número de clusters')
    parser.add_argument('--output', type=str, required=True, help='Nome base dos arquivos (ex: dados_pequeno)')
    parser.add_argument('--seed', type=int, default=42, help='Seed para reprodutibilidade')
    
    args = parser.parse_args()
    
    if args.N <= 0 or args.K <= 0:
        print("Erro: N e K devem ser positivos", file=sys.stderr)
        sys.exit(1)
    
    if args.K > args.N:
        print("Erro: K não pode ser maior que N", file=sys.stderr)
        sys.exit(1)
    
    print(f"Gerando {args.N} pontos em {args.K} clusters...")
    
    # Gera dados
    data, true_centroids = generate_kmeans_data(args.N, args.K, args.seed)
    
    # Gera centróides iniciais aleatórios
    data_range = (data.min(), data.max())
    initial_centroids = generate_initial_centroids(args.K, data_range, args.seed)
    
    # Salva arquivos
    data_file = f"{args.output}.csv"
    centroids_file = f"{args.output}_centroides_init.csv"
    true_centroids_file = f"{args.output}_centroides_true.csv"
    
    save_csv_1col(data_file, data)
    save_csv_1col(centroids_file, initial_centroids)
    save_csv_1col(true_centroids_file, true_centroids)
    
    print(f"✓ Dados salvos em: {data_file}")
    print(f"✓ Centróides iniciais: {centroids_file}")
    print(f"✓ Centróides verdadeiros: {true_centroids_file}")
    print(f"\nEstatísticas:")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  Média: {data.mean():.2f}")
    print(f"  Desvio padrão: {data.std():.2f}")

if __name__ == "__main__":
    main()
