# K-Means 1D - OpenMP Implementation

## Descrição

Implementação de K-means 1D usando **OpenMP** para paralelização em memória compartilhada (CPU multi-core).

## Características

- Assignment: paralelizado com `#pragma omp parallel for` e `reduction(+:sse)`
- Update: acumuladores por thread para evitar contenção
- Configurável: número de threads, scheduling strategy, chunk size

## Compilação

\`\`\`bash
gcc -O2 -std=c99 -fopenmp kmeans_1d_parallel.c -o kmeans_1d_parallel -lm
\`\`\`

Ou usando o Makefile:
\`\`\`bash
make
\`\`\`

## Execução

\`\`\`bash
./kmeans_1d_parallel dados.csv centroides_iniciais.csv [max_iter] [eps] [threads] [schedule] [chunk] [assign.csv] [centroids.csv]
\`\`\`

### Parâmetros:
- `dados.csv`: arquivo CSV com N pontos
- `centroides_iniciais.csv`: arquivo CSV com K centróides iniciais
- `max_iter`: número máximo de iterações (padrão: 50)
- `eps`: critério de convergência (padrão: 1e-4)
- `threads`: número de threads OpenMP (0 = todos os cores, padrão: 0)
- `schedule`: estratégia de scheduling (static, dynamic, guided, padrão: static)
- `chunk`: tamanho do chunk (0 = automático, padrão: 0)
- `assign.csv`: arquivo de saída com assignments (opcional)
- `centroids.csv`: arquivo de saída com centróides finais (opcional)

### Exemplos:
\`\`\`bash
# Usar todos os cores disponíveis
./kmeans_1d_parallel dados.csv centroides.csv

# 4 threads, scheduling estático
./kmeans_1d_parallel dados.csv centroides.csv 50 1e-6 4 static 0

# 8 threads, scheduling dinâmico, chunk=10
./kmeans_1d_parallel dados.csv centroides.csv 50 1e-6 8 dynamic 10

# 16 threads, guided scheduling
./kmeans_1d_parallel dados.csv centroides.csv 50 1e-6 16 guided 0
\`\`\`

## Scheduling Strategies

- **static**: Divide iterações igualmente entre threads no início
  - Melhor para carga balanceada
  - Menor overhead
  
- **dynamic**: Distribui iterações dinamicamente em runtime
  - Melhor para carga desbalanceada
  - Maior overhead
  
- **guided**: Similar a dynamic, mas com chunks que diminuem de tamanho
  - Compromisso entre static e dynamic

## Performance

**Esperado:**
- Speedup próximo de T para T threads (T pequeno: 2-8)
- Eficiência > 80% para até 8 threads
- Diminishing returns para T > número de cores físicos

**Benchmarks sugeridos:**
\`\`\`bash
for T in 1 2 4 8 16; do
  ./kmeans_1d_parallel dados.csv centroides.csv 50 1e-6 $T static 0
done
\`\`\`
\`\`\`

```makefile file="cuda/Makefile"
# Makefile para K-means CUDA

NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_70
LDFLAGS = -lm

TARGET = kmeans_1d_cuda
SRC = kmeans_1d_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
