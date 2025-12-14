#!/usr/bin/env python3
import numpy as np
import sys

def gerar_dados_1d(N, K, nome_dados, nome_centroides, seed=42):
    np.random.seed(seed)
    centroides_reais = np.linspace(0, 100, K)
    pontos_por_cluster = N // K

    dados = []
    for centro in centroides_reais:
        dados.extend(np.random.normal(centro, 2.0, pontos_por_cluster))

    resto = N - len(dados)
    if resto > 0:
        dados.extend(np.random.normal(centroides_reais[-1], 2.0, resto))

    dados = np.array(dados)
    np.random.shuffle(dados)
    np.savetxt(nome_dados, dados, fmt='%.6f')

    centroides_iniciais = centroides_reais + np.random.uniform(-5, 5, K)
    np.savetxt(nome_centroides, centroides_iniciais, fmt='%.6f')

    print(f"✓ {N:,} pontos, {K} clusters → {nome_dados}")

if __name__ == "__main__":
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    gerar_dados_1d(N, K, sys.argv[3], sys.argv[4], 42)
