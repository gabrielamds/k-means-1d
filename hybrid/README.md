# K-Means 1D - Hybrid Implementations

## Descrição

Implementações híbridas combinando múltiplos paradigmas de paralelização para explorar diferentes níveis de hierarquia computacional.

## Versões Disponíveis

### 1. OpenMP + CUDA (`kmeans_1d_omp_cuda.cu`)

**Arquitetura:** CPU multi-threaded + GPU acceleration

**Estratégia:**
- Divide dados em chunks entre threads OpenMP
- Cada thread OpenMP coordena operações CUDA assincronamente
- Usa `cudaStreamCreate` para overlapping de operações GPU
- Ideal para sistemas com CPU multi-core + GPU única

**Compilação:**
\`\`\`bash
nvcc -O2 -Xcompiler -fopenmp kmeans_1d_omp_cuda.cu -o kmeans_1d_omp_cuda -lm
\`\`\`

**Execução:**
\`\`\`bash
./kmeans_1d_omp_cuda dados.csv centroides.csv 50 1e-6 4 256
# 4 threads OpenMP, 256 threads por bloco CUDA
\`\`\`

**Benefícios:**
- Explora CPU e GPU em paralelo
- Reduz idle time da GPU via streams assíncronos
- Escalável com número de threads OpenMP

---

### 2. OpenMP + MPI (`kmeans_1d_omp_mpi.c`)

**Arquitetura:** Distributed memory (MPI) + Shared memory (OpenMP)

**Estratégia:**
- MPI distribui dados entre nós (inter-node parallelism)
- OpenMP paraleliza dentro de cada nó (intra-node parallelism)
- Exemplo: 2 processos MPI × 4 threads OpenMP = 8 workers totais

**Compilação:**
\`\`\`bash
mpicc -O2 -std=c99 -fopenmp kmeans_1d_omp_mpi.c -o kmeans_1d_omp_mpi -lm
\`\`\`

**Execução:**
\`\`\`bash
mpirun -np 2 ./kmeans_1d_omp_mpi dados.csv centroides.csv 50 1e-6 4
# 2 processos MPI, 4 threads OpenMP cada = 8 workers totais
\`\`\`

**Benefícios:**
- Escalabilidade em clusters multi-node
- Melhor uso de recursos NUMA
- Reduz overhead de comunicação MPI vs MPI puro

---

### 3. MPI + CUDA (`kmeans_1d_mpi_cuda.cu`)

**Arquitetura:** Distributed GPUs

**Estratégia:**
- Cada processo MPI controla uma GPU diferente
- Rank 0 → GPU 0, Rank 1 → GPU 1, etc.
- MPI distribui dados entre processos/GPUs
- MPI_Allreduce sincroniza centróides entre GPUs

**Compilação:**
\`\`\`bash
nvcc -O2 -arch=sm_70 kmeans_1d_mpi_cuda.cu -o kmeans_1d_mpi_cuda -lmpi
\`\`\`

**Execução:**
\`\`\`bash
mpirun -np 4 ./kmeans_1d_mpi_cuda dados.csv centroides.csv 50 1e-6 256
# 4 processos MPI, cada um usando 1 GPU, 256 threads/bloco
\`\`\`

**Benefícios:**
- Escalabilidade multi-GPU (sistemas HPC)
- Cada GPU processa N/P pontos
- Combina poder computacional de múltiplas GPUs

---

## Comparação de Performance

| Híbrido | Use Case | Speedup Esperado | Overhead Principal |
|---------|----------|------------------|-------------------|
| **OpenMP + CUDA** | Workstation com CPU forte + 1 GPU | 1.5-2x vs CUDA puro | Sincronização threads |
| **OpenMP + MPI** | Cluster multi-node | 1.2-1.5x vs MPI puro | Comunicação MPI |
| **MPI + CUDA** | Sistema multi-GPU (HPC) | Linear com # GPUs | Comunicação inter-GPU |

## Trade-offs

### OpenMP + CUDA
- **Prós:** Melhor utilização de CPU+GPU, overlapping de operações
- **Contras:** Complexidade de código, gerenciamento de streams
- **Quando usar:** 1 GPU potente + CPU multi-core idle

### OpenMP + MPI
- **Prós:** Escalável em clusters, menos comunicação que MPI puro
- **Contras:** Requer ambiente MPI + OpenMP configurado
- **Quando usar:** Clusters com nós multi-core

### MPI + CUDA
- **Prós:** Escalabilidade multi-GPU, datasets massivos
- **Contras:** Requer múltiplas GPUs, overhead de comunicação inter-GPU
- **Quando usar:** Sistemas HPC com múltiplas GPUs disponíveis

## Benchmarks Recomendados

Para cada implementação híbrida, teste com:

1. **Variação de recursos:**
   - OpenMP+CUDA: 1, 2, 4 threads × block sizes 128, 256
   - OpenMP+MPI: (1,2,4 threads) × (2,4 processos MPI)
   - MPI+CUDA: 2, 4 processos × block sizes 256, 512

2. **Tamanhos de dados:**
   - Pequeno (N=10^4): overhead pode dominar
   - Médio (N=10^5): sweet spot para híbridos
   - Grande (N=10^6): vantagem clara de híbridos

3. **Métricas:**
   - Tempo total vs componentes individuais
   - Speedup vs versões puras (OpenMP, MPI, CUDA)
   - Eficiência (speedup / # recursos)
   - Overhead de sincronização/comunicação
