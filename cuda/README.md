# K-Means 1D - CUDA Implementation

## Descrição

Implementação de K-means 1D usando **CUDA** para aceleração em GPU.

## Características

- Assignment: kernel paralelo (1 thread por ponto)
- Update: executado no host (CPU)
- Métricas detalhadas: tempo H2D, kernel, D2H separadamente
- Configurável: block size (128, 256, 512)

## Compilação

\`\`\`bash
nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda -lm
\`\`\`

Ou usando o Makefile:
\`\`\`bash
make
\`\`\`

## Execução

\`\`\`bash
./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter] [eps] [block_size] [assign.csv] [centroids.csv]
\`\`\`

### Parâmetros:
- `dados.csv`: arquivo CSV com N pontos
- `centroides_iniciais.csv`: arquivo CSV com K centróides iniciais
- `max_iter`: número máximo de iterações (padrão: 50)
- `eps`: critério de convergência (padrão: 1e-4)
- `block_size`: threads por bloco (128, 256, ou 512, padrão: 256)
- `assign.csv`: arquivo de saída com assignments (opcional)
- `centroids.csv`: arquivo de saída com centróides finais (opcional)

### Exemplos:
\`\`\`bash
# Block size padrão (256)
./kmeans_1d_cuda dados.csv centroides.csv

# Block size 128
./kmeans_1d_cuda dados.csv centroides.csv 50 1e-6 128

# Block size 512
./kmeans_1d_cuda dados.csv centroides.csv 50 1e-6 512
\`\`\`

## Block Size

O block size afeta a ocupação da GPU e a performance:

- **128**: Menor ocupação, mais flexibilidade de registradores
- **256**: Padrão, bom compromisso geral
- **512**: Maior ocupação, pode ter contenção de recursos

**Benchmarks sugeridos:**
\`\`\`bash
for BS in 128 256 512; do
  ./kmeans_1d_cuda dados.csv centroides.csv 50 1e-6 $BS
done
\`\`\`

## Métricas de Performance

A implementação reporta:
- **H2D (Host→Device)**: Tempo de transferência CPU→GPU
- **Kernel (Assignment)**: Tempo de execução do kernel
- **D2H (Device→Host)**: Tempo de transferência GPU→CPU
- **Throughput**: Milhões de pontos processados por segundo

## Requisitos

- CUDA Toolkit instalado
- GPU compatível com CUDA
- Driver NVIDIA atualizado

Verificar GPU disponível:
\`\`\`bash
nvidia-smi
\`\`\`
