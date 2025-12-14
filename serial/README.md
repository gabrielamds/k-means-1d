# K-Means 1D - Serial Implementation (Baseline)

## Descrição

Implementação sequencial (naive) de K-means 1D. Esta versão serve como **baseline** para comparação de performance com as versões paralelas.

## Características

- Sem paralelização
- Assignment: loop sequencial sobre N pontos
- Update: acumulação sequencial
- Referência para cálculo de speedup

## Compilação

\`\`\`bash
gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
\`\`\`

Ou usando o Makefile:
\`\`\`bash
make
\`\`\`

## Execução

\`\`\`bash
./kmeans_1d_naive dados.csv centroides_iniciais.csv [max_iter] [eps] [assign.csv] [centroids.csv]
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
./kmeans_1d_naive ../data/dados_pequeno.csv ../data/centroides_init.csv 50 1e-6
\`\`\`

## Performance

Esta é a versão mais lenta, usada como baseline. Todas as outras implementações (OpenMP, CUDA, MPI, híbridos) devem ter speedup > 1.0 comparado a esta versão.

## Formato dos Arquivos CSV

- Sem cabeçalho
- Um número por linha
- Aceita vírgula, ponto-e-vírgula, espaço ou tab como delimitadores
\`\`\`

```makefile file="openmp/Makefile"
# Makefile para K-means OpenMP

CC = gcc
CFLAGS = -O2 -std=c99 -fopenmp -Wall
LDFLAGS = -lm

TARGET = kmeans_1d_parallel
SRC = kmeans_1d_parallel.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
