# Guia de Uso e Análise - K-means 1D

Este diretório contém instruções detalhadas de uso, análise de resultados e relatórios gerados.

## Estrutura

\`\`\`
report/
├── README.md              # Este arquivo
├── RESULTS.md             # Análise completa de resultados (gerado)
├── HYBRID_ANALYSIS.md     # Análise específica de híbridos
├── EFFICIENCY_ANALYSIS.md # Análise de eficiência (gerado)
└── figures/               # Gráficos gerados
    ├── speedup_*.png
    ├── efficiency_*.png
    └── *_scaling.png
\`\`\`

## Como Reproduzir os Resultados

### Passo 1: Preparar Ambiente

Certifique-se de ter todos os requisitos instalados:

\`\`\`bash
# Verificar compiladores
gcc --version          # >= 7.0
nvcc --version         # >= 11.0 (opcional)
mpicc --version        # >= 4.0 (opcional)

# Verificar Python
python3 --version      # >= 3.8
pip3 install matplotlib numpy
\`\`\`

### Passo 2: Compilar Tudo

\`\`\`bash
cd <raiz-do-projeto>
make compile-all
\`\`\`

Isso compila:
- Serial (`serial/kmeans_1d_naive`)
- OpenMP (`openmp/kmeans_1d_parallel`)
- CUDA (`cuda/kmeans_1d_cuda`) - se disponível
- MPI (`mpi/kmeans_1d_mpi`) - se disponível
- Híbridos (`hybrid/*`) - conforme disponibilidade

### Passo 3: Gerar Dados

\`\`\`bash
make data
\`\`\`

Ou manualmente:

\`\`\`bash
cd data
python3 generate_data.py --N 10000 --K 4 --output dados_pequeno
python3 generate_data.py --N 100000 --K 8 --output dados_medio
python3 generate_data.py --N 1000000 --K 16 --output dados_grande
\`\`\`

### Passo 4: Executar Benchmarks

#### Opção A: Tudo de uma vez

\`\`\`bash
make benchmark-all
\`\`\`

Isso executa todos os benchmarks e salva em `results/`.

#### Opção B: Individual

\`\`\`bash
# Serial
bash scripts/benchmark_serial.sh > results/serial_results.txt

# OpenMP
bash scripts/benchmark_openmp.sh > results/openmp_results.txt

# CUDA
bash scripts/benchmark_cuda.sh > results/cuda_results.txt

# MPI
bash scripts/benchmark_mpi.sh > results/mpi_results.txt

# Híbridos
bash scripts/benchmark_hybrid.sh > results/hybrid_results.txt
\`\`\`

### Passo 5: Gerar Análises

\`\`\`bash
# Gera todos os gráficos e relatórios
make analyze
\`\`\`

Ou individualmente:

\`\`\`bash
python3 analysis/analyze_speedup.py
python3 analysis/analyze_scalability.py
python3 analysis/analyze_efficiency.py
python3 analysis/generate_report.py
\`\`\`

## Interpretação dos Resultados

### Speedup

**Fórmula:** Speedup = T_serial / T_parallel

**Interpretação:**
- Speedup = 1: Sem ganho de performance
- Speedup = P: Speedup linear (ideal)
- Speedup < P: Overhead reduz eficiência
- Speedup > P: Super-linear (raro, cache effects)

**Gráficos:**
- X: Número de recursos (threads, processos)
- Y: Speedup
- Linha ideal: y = x (speedup linear)

### Eficiência

**Fórmula:** Eficiência = Speedup / P × 100%

**Interpretação:**
- Eficiência = 100%: Uso perfeito de recursos
- Eficiência >= 80%: Bom
- Eficiência >= 50%: Aceitável
- Eficiência < 50%: Overhead excessivo

**Thresholds:**
- Verde (100%): Ideal
- Laranja (80%): Threshold aceitável
- Vermelho (50%): Baixa eficiência

### Escalabilidade

**Strong Scaling:**
- Problema de tamanho fixo
- Aumenta número de recursos
- Esperado: Tempo diminui proporcionalmente

**Weak Scaling:**
- Carga por recurso constante (N/P constante)
- Aumenta N e P proporcionalmente
- Esperado: Tempo constante

## Comparação de Paradigmas

### OpenMP

**Prós:**
- Fácil de implementar
- Overhead mínimo
- Bom speedup até 8-16 threads

**Contras:**
- Limitado a 1 máquina
- Escalabilidade limitada por # cores

**Melhor para:**
- Workstations multi-core
- Datasets médios a grandes
- Desenvolvimento rápido

### CUDA

**Prós:**
- Speedup excelente para N grande
- Milhares de threads paralelas
- Alto throughput

**Contras:**
- Overhead H2D/D2H
- Complexidade maior
- Requer GPU

**Melhor para:**
- Datasets grandes (N > 100k)
- Sistemas com GPU potente
- Computação massivamente paralela

### MPI

**Prós:**
- Escalável para múltiplos nós
- Memória distribuída
- Flexível

**Contras:**
- Overhead de comunicação
- Mais complexo
- Speedup moderado

**Melhor para:**
- Clusters HPC
- Datasets massivos
- Memória distribuída necessária

### Híbridos

**OpenMP + CUDA:**
- Usa CPU e GPU simultaneamente
- Melhor utilização de recursos
- Speedup adicional de 20-30%

**OpenMP + MPI:**
- Inter-node (MPI) + intra-node (OpenMP)
- Reduz comunicação vs MPI puro
- Ideal para clusters com nós multi-core

**MPI + CUDA:**
- Multi-GPU distribuído
- Escalabilidade para múltiplas GPUs
- Ideal para HPC com múltiplas GPUs

## Benchmarks Customizados

### Testar Diferentes Configurações OpenMP

\`\`\`bash
# Variar threads
for T in 1 2 4 8 16; do
  ./openmp/kmeans_1d_parallel dados.csv centroides.csv 50 1e-6 $T static 0
done

# Variar scheduling
for SCHED in static dynamic guided; do
  ./openmp/kmeans_1d_parallel dados.csv centroides.csv 50 1e-6 8 $SCHED 0
done

# Variar chunk size (dynamic)
for CHUNK in 1 10 100 1000; do
  ./openmp/kmeans_1d_parallel dados.csv centroides.csv 50 1e-6 8 dynamic $CHUNK
done
\`\`\`

### Testar Diferentes Block Sizes CUDA

\`\`\`bash
for BS in 128 256 512; do
  ./cuda/kmeans_1d_cuda dados.csv centroides.csv 50 1e-6 $BS
done
\`\`\`

### Testar Escalabilidade MPI

\`\`\`bash
for NP in 1 2 4 8 16; do
  mpirun -np $NP ./mpi/kmeans_1d_mpi dados.csv centroides.csv 50 1e-6
done
\`\`\`

## Gerar Relatório Customizado

Você pode modificar `analysis/generate_report.py` para incluir:

1. Datasets específicos
2. Configurações específicas
3. Comparações customizadas
4. Análises adicionais

## Ambiente de Teste Recomendado

### Workstation

- CPU: 8+ cores (Intel i7/i9, AMD Ryzen)
- RAM: 16GB+
- GPU: NVIDIA RTX 3060+ (opcional)
- SO: Linux (Ubuntu 20.04+)

### Cluster

- Nós: 4+ nós com 8+ cores cada
- Rede: Infiniband ou 10GbE
- MPI: Open MPI 4.0+
- SO: Linux (CentOS/RHEL)

### HPC

- Nós com múltiplas GPUs
- MPI + CUDA
- Scheduler: SLURM/PBS

## Solução de Problemas

### Resultados inconsistentes

\`\`\`bash
# Rodar múltiplas vezes e calcular média
for i in {1..5}; do
  ./serial/kmeans_1d_naive dados.csv centroides.csv >> results.txt
done
\`\`\`

### SSE diverge entre versões

- Verifique centróides iniciais (mesmos para todas versões)
- Verifique precisão numérica (double vs float)
- Verifique ordenação de operações paralelas

### Performance pior que serial

- Dataset muito pequeno (overhead domina)
- Configuração inadequada (muitos threads/processos)
- Hardware inadequado (CPU fraca, GPU antiga)

## Contato e Suporte

Para questões sobre uso, análise ou interpretação dos resultados, consulte:

1. Este README
2. `../README.md` principal
3. Comentários no código-fonte
4. Relatórios gerados em `RESULTS.md`
