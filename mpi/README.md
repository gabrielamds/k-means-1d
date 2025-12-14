# K-Means 1D - MPI Implementation

## Descrição

Implementação de K-means 1D usando **MPI (Message Passing Interface)** para computação distribuída em memória distribuída.

## Características

- Distribuição de dados entre P processos MPI
- Assignment paralelo: cada processo calcula para sua partição local
- Update distribuído: `MPI_Allreduce` para somar SSE, sums e counts globalmente
- Broadcast de centróides com `MPI_Bcast` no início de cada iteração
- Medição separada de tempo de comunicação

## Compilação

\`\`\`bash
mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
\`\`\`

Ou usando o Makefile:
\`\`\`bash
make
\`\`\`

## Execução

\`\`\`bash
mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter] [eps] [assign.csv] [centroids.csv]
\`\`\`

### Parâmetros:
- `dados.csv`: arquivo CSV com N pontos (1 valor por linha)
- `centroides_iniciais.csv`: arquivo CSV com K centróides iniciais
- `max_iter`: número máximo de iterações (padrão: 50)
- `eps`: critério de convergência (padrão: 1e-4)
- `assign.csv`: arquivo de saída com assignments (opcional)
- `centroids.csv`: arquivo de saída com centróides finais (opcional)

### Exemplo:
\`\`\`bash
mpirun -np 8 ./kmeans_1d_mpi ../data/dados_grande.csv ../data/centroides_init.csv 50 1e-6
\`\`\`

## Variação do Número de Processos

Para análise de escalabilidade, teste com diferentes números de processos:

\`\`\`bash
mpirun -np 1 ./kmeans_1d_mpi dados.csv centroides.csv
mpirun -np 2 ./kmeans_1d_mpi dados.csv centroides.csv
mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides.csv
mpirun -np 8 ./kmeans_1d_mpi dados.csv centroides.csv
mpirun -np 16 ./kmeans_1d_mpi dados.csv centroides.csv
\`\`\`

## Métricas

A implementação mede:
- **Tempo total**: tempo de execução completo
- **Tempo de comunicação**: tempo gasto em `MPI_Allreduce`, `MPI_Bcast`, `MPI_Scatterv` e `MPI_Gatherv`
- **Speedup**: calculado vs versão sequencial
- **Eficiência**: speedup / número de processos

## Arquitetura

### Distribuição de Dados
- Rank 0 lê o arquivo completo
- `MPI_Scatterv` distribui dados entre processos
- Cada processo mantém N/P pontos localmente

### Assignment Step
- Cada processo calcula assignment apenas para seus N/P pontos
- SSE local é somado via `MPI_Allreduce(&sse_local, &sse_global, ...)`

### Update Step
- Cada processo acumula sums e counts localmente
- `MPI_Allreduce` soma acumuladores de todos os processos
- Todos os processos calculam os novos centróides (garantindo consistência)

### Coleta de Resultados
- `MPI_Gatherv` coleta assignments de volta ao rank 0
- Rank 0 salva os arquivos de saída

## Performance

**Esperado:**
- Speedup próximo de P para P pequeno (2-8 processos)
- Overhead de comunicação aumenta com P
- Ponto de diminishing returns quando comunicação domina

**Trade-offs:**
- Menos memória por processo (N/P vs N)
- Overhead de comunicação (`MPI_Allreduce`, `MPI_Bcast`)
- Escalabilidade para grandes N e clusters multi-node
