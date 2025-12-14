// kmeans_1d_cuda.cu - Versão CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// ==================== MACROS DE VERIFICAÇÃO DE ERROS ====================

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error em %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ==================== FUNÇÕES AUXILIARES CSV (HOST) ====================

int count_rows(const char *filename) {
    FILE *f = fopen(filename, "r");
    if(!f) {
        fprintf(stderr, "Erro ao abrir %s\n", filename);
        return 0;
    }
    int count = 0;
    char line[256];
    while(fgets(line, sizeof(line), f)) {
        count++;
    }
    fclose(f);
    return count;
}

double* read_csv_1col(const char *filename, int *n) {
    *n = count_rows(filename);
    if(*n == 0) return NULL;
    
    double *arr = (double*)malloc((*n) * sizeof(double));
    FILE *f = fopen(filename, "r");
    for(int i=0; i<*n; i++) {
        fscanf(f, "%lf", &arr[i]);
    }
    fclose(f);
    return arr;
}

void write_assign_csv(const char *filename, const int *assign, int n) {
    FILE *f = fopen(filename, "w");
    for(int i=0; i<n; i++) {
        fprintf(f, "%d\n", assign[i]);
    }
    fclose(f);
}

void write_centroids_csv(const char *filename, const double *C, int k) {
    FILE *f = fopen(filename, "w");
    for(int c=0; c<k; c++) {
        fprintf(f, "%.10lf\n", C[c]);
    }
    fclose(f);
}

// ==================== UPDATE STEP (CPU) ====================

void update_step_1d(const double *X, double *C, const int *assign, int N, int K) {
    double *sum = (double*)calloc(K, sizeof(double));
    int *cnt = (int*)calloc(K, sizeof(int));
    
    for(int i=0; i<N; i++) {
        int a = assign[i];
        cnt[a]++;
        sum[a] += X[i];
    }
    
    for(int c=0; c<K; c++) {
        if(cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = X[0];
    }
    
    free(sum);
    free(cnt);
}

// ==================== KERNEL CUDA - ASSIGNMENT ====================

__global__ void assignment_kernel(const double *X, const double *C, 
                                   int *assign, double *errors, 
                                   int N, int K) {
    // Calcular índice global da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Verificar se está dentro dos limites
    if(i < N) {
        int best = -1;
        double bestd = 1e300;
        
        // Encontrar o centróide mais próximo
        for(int c=0; c<K; c++) {
            double diff = X[i] - C[c];
            double d = diff * diff;
            if(d < bestd) {
                bestd = d;
                best = c;
            }
        }
        
        // Armazenar resultados
        assign[i] = best;
        errors[i] = bestd;
    }
}

// ==================== FUNÇÃO K-MEANS CUDA ====================

void kmeans_1d_cuda(const double *X_host, double *C_host, int *assign_host,
                    int N, int K, int max_iter, double eps,
                    int threadsPerBlock) {
    
    // ========== ALOCAÇÃO DE MEMÓRIA NA GPU ==========
    double *X_dev, *C_dev, *errors_dev;
    int *assign_dev;
    
    CUDA_CHECK(cudaMalloc(&X_dev, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&C_dev, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&assign_dev, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&errors_dev, N * sizeof(double)));
    
    // ========== COPIAR DADOS PARA GPU ==========
    CUDA_CHECK(cudaMemcpy(X_dev, X_host, N * sizeof(double), cudaMemcpyHostToDevice));
    
    // ========== CONFIGURAÇÃO DO KERNEL ==========
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("\nConfiguração CUDA:\n");
    printf("  Threads por bloco: %d\n", threadsPerBlock);
    printf("  Número de blocos: %d\n", numBlocks);
    printf("  Total de threads: %d\n", numBlocks * threadsPerBlock);
    
    // ========== LOOP K-MEANS ==========
    double prev_sse = 1e300;
    double *errors_host = (double*)malloc(N * sizeof(double));
    
    printf("\nIteração | SSE\n");
    printf("---------|---------------\n");
    
    for(int it=0; it<max_iter; it++) {
        // Copiar centróides atuais para GPU
        CUDA_CHECK(cudaMemcpy(C_dev, C_host, K * sizeof(double), cudaMemcpyHostToDevice));
        
        // ASSIGNMENT NA GPU
        assignment_kernel<<<numBlocks, threadsPerBlock>>>(X_dev, C_dev, 
                                                           assign_dev, errors_dev, 
                                                           N, K);
        
        // Verificar erros de kernel
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copiar resultados para CPU
        CUDA_CHECK(cudaMemcpy(assign_host, assign_dev, N * sizeof(int), 
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(errors_host, errors_dev, N * sizeof(double), 
                              cudaMemcpyDeviceToHost));
        
        // Calcular SSE
        double sse = 0.0;
        for(int i=0; i<N; i++) {
            sse += errors_host[i];
        }
        
        printf("%8d | %.6lf\n", it, sse);
        
        // Verificar convergência
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps) {
            printf("\nConvergiu após %d iterações!\n", it+1);
            break;
        }
        
        // UPDATE NA CPU
        update_step_1d(X_host, C_host, assign_host, N, K);
        prev_sse = sse;
    }
    
    // ========== LIBERAR MEMÓRIA ==========
    free(errors_host);
    CUDA_CHECK(cudaFree(X_dev));
    CUDA_CHECK(cudaFree(C_dev));
    CUDA_CHECK(cudaFree(assign_dev));
    CUDA_CHECK(cudaFree(errors_dev));
}

// ==================== MAIN ====================

int main(int argc, char **argv) {
    if(argc < 8) {
        printf("Uso: %s dados.csv centroides.csv max_iter eps threads assign.csv centroids.csv\n", argv[0]);
        printf("  threads: número de threads por bloco (ex: 256)\n");
        return 1;
    }
    
    const char *dados_file = argv[1];
    const char *centroides_file = argv[2];
    int max_iter = atoi(argv[3]);
    double eps = atof(argv[4]);
    int threadsPerBlock = atoi(argv[5]);
    const char *assign_out = argv[6];
    const char *centroids_out = argv[7];
    
    // Carregar dados
    int N, K;
    double *X = read_csv_1col(dados_file, &N);
    double *C = read_csv_1col(centroides_file, &K);
    int *assign = (int*)malloc(N * sizeof(int));
    
    printf("=== K-MEANS 1D CUDA ===\n");
    printf("N = %d pontos, K = %d clusters\n", N, K);
    
    // Criar eventos CUDA para medição de tempo
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Iniciar cronômetro
    CUDA_CHECK(cudaEventRecord(start));
    
    // Executar K-means CUDA
    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, threadsPerBlock);
    
    // Parar cronômetro
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float tempo_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tempo_ms, start, stop));
    
    printf("\nTempo de execução: %.2f ms\n", tempo_ms);
    
    // Salvar resultados
    write_assign_csv(assign_out, assign, N);
    write_centroids_csv(centroids_out, C, K);
    
    printf("\nResultados salvos em %s e %s\n", assign_out, centroids_out);
    
    // Liberar memória e eventos
    free(X);
    free(C);
    free(assign);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
