# K-means 1D - Projeto de Programação Concorrente e Distribuída

Implementação completa do algoritmo K-means 1D com múltiplos paradigmas de paralelização: Serial, OpenMP, CUDA, MPI e implementações híbridas.

## Estrutura do Projeto

\`\`\`
projeto-pcd-kmeans/
├── serial/              # Implementação sequencial (baseline)
├── openmp/              # Paralelização OpenMP (memória compartilhada)
├── cuda/                # Aceleração GPU com CUDA
├── mpi/                 # Computação distribuída com MPI
├── hybrid/              # Implementações híbridas
│   ├── kmeans_1d_omp_cuda.cu      # OpenMP + CUDA
│   ├── kmeans_1d_omp_mpi.c        # OpenMP + MPI
│   └── kmeans_1d_mpi_cuda.cu      # MPI + CUDA
├── data/                # Datasets e scripts de geração
├── scripts/             # Scripts de benchmark
├── analysis/            # Scripts de análise Python
├── report/              # Relatórios e gráficos gerados
└── results/             # Resultados de benchmarks
\`\`\`

## Requisitos

### Compiladores e Bibliotecas

- **GCC** 7.0+ com suporte a C99 e OpenMP
- **CUDA Toolkit** 11.0+ (opcional, para versões CUDA)
- **Open MPI** 4.0+ (opcional, para versões MPI)
- **Python** 3.8+ com matplotlib e numpy (para análises)

### Verificar Instalação

\`\`\`bash
# GCC e OpenMP
gcc --version
gcc -fopenmp --version

# CUDA (se disponível)
nvcc --version
nvidia-smi

# MPI (se disponível)
mpicc --version
mpirun --version

# Python
python3 --version
pip3 install matplotlib numpy
\`\`\`

## Início Rápido

### 1. Compilar Tudo

\`\`\`bash
make compile-all
\`\`\`

Isso compila todas as versões disponíveis no sistema.

### 2. Gerar Dados de Teste

\`\`\`bash
make data
\`\`\`

Gera datasets sintéticos:
- `dados_pequeno.csv` (10k pontos, 4 clusters)
- `dados_medio.csv` (100k pontos, 8 clusters)
- `dados_grande.csv` (1M pontos, 16 clusters)

### 3. Executar Benchmarks

\`\`\`bash
make benchmark-all
\`\`\`

Roda todos os benchmarks e salva resultados em `results/`.

### 4. Gerar Análises

\`\`\`bash
make analyze
\`\`\`

Gera gráficos de speedup, escalabilidade e eficiência em `report/figures/`.

## Uso Individual

### Serial (Baseline)

\`\`\`bash
cd serial
make
./kmeans_1d_naive ../data/dados_medio.csv ../data/dados_medio_centroides_init.csv 50 1e-6
\`\`\`

### OpenMP

\`\`\`bash
cd openmp
make
./kmeans_1d_parallel ../data/dados_medio.csv ../data/dados_medio_centroides_init.csv 50 1e-6 8 static 0
# Parâmetros: max_iter eps threads schedule chunk
\`\`\`

### CUDA

\`\`\`bash
cd cuda
make
./kmeans_1d_cuda ../data/dados_grande.csv ../data/dados_grande_centroides_init.csv 50 1e-6 256
# Parâmetros: max_iter eps block_size
\`\`\`

### MPI

\`\`\`bash
cd mpi
make
mpirun -np 4 ./kmeans_1d_mpi ../data/dados_grande.csv ../data/dados_grande_centroides_init.csv 50 1e-6
# -np: número de processos
\`\`\`

### Híbridos

\`\`\`bash
cd hybrid
make

# OpenMP + CUDA
./kmeans_1d_omp_cuda ../data/dados_medio.csv ../data/dados_medio_centroides_init.csv 50 1e-6 4 256

# OpenMP + MPI
mpirun -np 2 ./kmeans_1d_omp_mpi ../data/dados_grande.csv ../data/dados_grande_centroides_init.csv 50 1e-6 4

# MPI + CUDA
mpirun -np 4 ./kmeans_1d_mpi_cuda ../data/dados_grande.csv ../data/dados_grande_centroides_init.csv 50 1e-6 256
\`\`\`

## Formato de Dados

Os arquivos CSV devem ter:
- **Sem cabeçalho**
- **Um valor por linha** (1D)
- Aceita delimitadores: vírgula, ponto-e-vírgula, espaço, tab

Exemplo (`dados.csv`):
\`\`\`
10.5
23.1
15.7
...
\`\`\`

## Parâmetros Comuns

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `max_iter` | Número máximo de iterações | 50 |
| `eps` | Critério de convergência (variação relativa SSE) | 1e-4 |
| `threads` | Número de threads OpenMP (0 = todos) | 0 |
| `schedule` | Scheduling OpenMP (static, dynamic, guided) | static |
| `chunk` | Chunk size OpenMP (0 = automático) | 0 |
| `block_size` | Threads por bloco CUDA (128, 256, 512) | 256 |

## Análise de Resultados

Os scripts de análise geram:

1. **Gráficos de Speedup** (`report/figures/speedup_*.png`)
   - Speedup vs número de recursos
   - Comparação entre paradigmas

2. **Escalabilidade** (`report/figures/*_scaling.png`)
   - Strong scaling (problema fixo)
   - Weak scaling (carga/recurso constante)

3. **Eficiência** (`report/figures/efficiency_*.png`)
   - Eficiência paralela (%)
   - Identificação de diminishing returns

4. **Relatório Completo** (`report/RESULTS.md`)
   - Análise detalhada
   - Recomendações
   - Trade-offs

## Resultados Esperados

### Speedup vs Serial

| Versão | Dataset Pequeno | Dataset Médio | Dataset Grande |
|--------|----------------|---------------|----------------|
| OpenMP (8 threads) | 6-7x | 7-8x | 7-8x |
| CUDA (256 blocks) | 5-8x | 10-12x | 12-15x |
| MPI (4 processos) | 3-4x | 3.5-4x | 4-4.5x |
| OpenMP+CUDA | 8-10x | 12-14x | 15-18x |

### Quando Usar Cada Paradigma

- **Serial**: Baseline, datasets pequenos, debugging
- **OpenMP**: CPU multi-core, datasets médios a grandes
- **CUDA**: GPU disponível, datasets grandes (N > 100k)
- **MPI**: Cluster multi-node, datasets massivos
- **OpenMP+CUDA**: Workstation forte (CPU+GPU)
- **OpenMP+MPI**: Cluster com nós multi-core
- **MPI+CUDA**: Sistema HPC multi-GPU

## Troubleshooting

### Compilação CUDA falha

\`\`\`bash
# Verificar CUDA instalado
nvcc --version

# Ajustar compute capability no Makefile
# Editar cuda/Makefile: -arch=sm_XX (consulte GPU)
\`\`\`

### MPI não encontrado

\`\`\`bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# CentOS/RHEL
sudo yum install openmpi openmpi-devel

# Adicionar ao PATH
export PATH=$PATH:/usr/lib64/openmpi/bin
\`\`\`

### Erros de memória (GPU)

\`\`\`bash
# Usar dataset menor ou reduzir block size
./kmeans_1d_cuda dados_pequeno.csv centroides.csv 50 1e-6 128
\`\`\`

## Limpeza

\`\`\`bash
# Remove executáveis
make clean

# Remove executáveis e dados gerados
make distclean
\`\`\`

## Contribuindo

Para adicionar novos paradigmas ou otimizações:

1. Crie diretório em `<paradigma>/`
2. Adicione `Makefile` e `README.md`
3. Mantenha interface consistente (argumentos CLI)
4. Adicione testes em `scripts/benchmark_<paradigma>.sh`
5. Documente no `README.md` principal

## Licença

Este projeto é material educacional para a disciplina de Programação Concorrente e Distribuída.

## Autores

Projeto desenvolvido como trabalho acadêmico de PCD.

## Referências

- Lloyd, S. (1982). "Least squares quantization in PCM"
- OpenMP Architecture Review Board. OpenMP Specification
- NVIDIA Corporation. CUDA C Programming Guide
- Message Passing Interface Forum. MPI Standard
