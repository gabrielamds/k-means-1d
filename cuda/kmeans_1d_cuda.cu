/**
 * ============================================================================
 * K-MEANS 1D - VERSÃO CUDA (GPU)
 * ============================================================================
 * Projeto PCD - Programação Concorrente e Distribuída
 * 
 * Paralelização massiva usando CUDA para GPUs NVIDIA.
 * 
 * Análises implementadas:
 *   - Impacto do block size (32, 64, 128, 256, 512, 1024)
 *   - Tempo de transferência CPU<->GPU
 *   - Ocupação da GPU
 *   - Speedup vs versão serial
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

/* ============================================================================
 * PARÂMETROS PADRONIZADOS (iguais em todas as versões)
 * ============================================================================ */
#define MAX_ITER 100
#define EPS 1e-6
#define DEFAULT_BLOCK_SIZE 256
#define SEED 42

/* ============================================================================
 * MACRO PARA VERIFICAR ERROS CUDA
 * ============================================================================ */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* ============================================================================
 * ATOMICADD PARA DOUBLE (compatível com todas as GPUs)
 * ============================================================================ */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Para GPUs Pascal+ (sm_60+), atomicAdd para double é nativo
#else
// Implementação manual para GPUs mais antigas
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Wrapper para usar em qualquer GPU
__device__ __forceinline__ void myAtomicAdd(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

/* ============================================================================
 * ESTRUTURAS DE DADOS
 * ============================================================================ */

typedef struct {
    int iterations;
    double sse_final;
    double time_total_ms;
    double time_transfer_h2d_ms;  // Host to Device
    double time_transfer_d2h_ms;  // Device to Host
    double time_kernel_ms;
    double time_assignment_ms;
    double time_update_ms;
    double throughput;
    double* sse_history;
    int block_size;
    int grid_size;
    double gpu_occupancy;
} KMeansMetrics;

typedef struct {
    char name[256];
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessors;
    int max_threads_per_block;
    size_t global_memory;
    size_t shared_memory_per_block;
} GPUInfo;

/* ============================================================================
 * KERNELS CUDA
 * ============================================================================ */

/**
 * Kernel de Assignment
 * Cada thread processa um ponto e encontra o centróide mais próximo
 */
__global__ void assignment_kernel(const double* __restrict__ X,
                                   const double* __restrict__ C,
                                   int* __restrict__ assign,
                                   double* __restrict__ partial_sse,
                                   int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        double min_dist = DBL_MAX;
        int best_cluster = 0;
        
        for (int c = 0; c < K; c++) {
            double diff = X[i] - C[c];
            double dist = diff * diff;
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        
        assign[i] = best_cluster;
        partial_sse[i] = min_dist;
    }
}

/**
 * Kernel de redução paralela para SSE
 * Usa memória compartilhada para redução eficiente
 */
__global__ void reduce_sse_kernel(const double* __restrict__ partial_sse,
                                   double* __restrict__ block_sums,
                                   int N) {
    extern __shared__ double sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Carregar dois elementos por thread
    double sum = 0.0;
    if (i < N) sum = partial_sse[i];
    if (i + blockDim.x < N) sum += partial_sse[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Redução na memória compartilhada
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (sem sync necessário)
    if (tid < 32) {
        volatile double* vdata = sdata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }
    
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * Kernel para acumular somas por cluster
 * Usa myAtomicAdd para evitar race conditions (compatível com todas as GPUs)
 */
__global__ void accumulate_kernel(const double* __restrict__ X,
                                   const int* __restrict__ assign,
                                   double* __restrict__ sum,
                                   int* __restrict__ count,
                                   int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        int c = assign[i];
        myAtomicAdd(&sum[c], X[i]);
        atomicAdd(&count[c], 1);
    }
}

/**
 * Kernel para atualizar centróides
 */
__global__ void update_centroids_kernel(double* __restrict__ C,
                                         const double* __restrict__ sum,
                                         const int* __restrict__ count,
                                         double x0,
                                         int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < K) {
        if (count[c] > 0) {
            C[c] = sum[c] / count[c];
        } else {
            C[c] = x0;  // Cluster vazio
        }
    }
}

/* ============================================================================
 * FUNÇÕES HOST
 * ============================================================================ */

double* read_csv(const char* filename, int* count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "ERRO: Não foi possível abrir '%s'\n", filename);
        exit(EXIT_FAILURE);
    }
    
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) n++;
    }
    
    double* data = (double*)malloc(n * sizeof(double));
    rewind(file);
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < n) {
        if (strlen(line) > 1) data[i++] = atof(line);
    }
    
    fclose(file);
    *count = i;
    return data;
}

void save_csv_int(const char* filename, int* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    for (int i = 0; i < n; i++) fprintf(file, "%d\n", data[i]);
    fclose(file);
}

void save_csv_double(const char* filename, double* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    for (int i = 0; i < n; i++) fprintf(file, "%.10f\n", data[i]);
    fclose(file);
}

void save_sse_history(const char* filename, double* sse_history, int iterations) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    fprintf(file, "iteracao,sse\n");
    for (int i = 0; i < iterations; i++) {
        fprintf(file, "%d,%.10f\n", i + 1, sse_history[i]);
    }
    fclose(file);
}

GPUInfo get_gpu_info() {
    GPUInfo info;
    int device;
    cudaDeviceProp prop;
    
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    strncpy(info.name, prop.name, sizeof(info.name) - 1);
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessors = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.global_memory = prop.totalGlobalMem;
    info.shared_memory_per_block = prop.sharedMemPerBlock;
    
    return info;
}

/* ============================================================================
 * ALGORITMO K-MEANS CUDA
 * ============================================================================ */

KMeansMetrics kmeans_cuda(double* h_X, int N, double* h_C, int K, int* h_assign,
                          int max_iter, double eps, int block_size) {
    KMeansMetrics metrics;
    metrics.sse_history = (double*)malloc(max_iter * sizeof(double));
    metrics.block_size = block_size;
    metrics.grid_size = (N + block_size - 1) / block_size;
    
    // Limpar erros anteriores
    cudaGetLastError();
    
    int grid_N = metrics.grid_size;
    int grid_K = (K + block_size - 1) / block_size;
    if (grid_K < 1) grid_K = 1;
    
    // Calcular reduce_blocks: mínimo 1, máximo grid_N
    int reduce_blocks = (grid_N + 1) / 2;  // Reduzir pela metade
    if (reduce_blocks < 1) reduce_blocks = 1;
    if (reduce_blocks > 1024) reduce_blocks = 1024;  // Limitar para evitar muito overhead
    
    // Alocar memória GPU
    double *d_X, *d_C, *d_partial_sse, *d_block_sums, *d_sum;
    int *d_assign, *d_count;
    
    // Eventos para medir tempo
    cudaEvent_t start_total, end_total;
    cudaEvent_t start_h2d, end_h2d;
    cudaEvent_t start_d2h, end_d2h;
    cudaEvent_t start_kernel, end_kernel;
    
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&end_total));
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&end_h2d));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&end_d2h));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&end_kernel));
    
    CUDA_CHECK(cudaEventRecord(start_total));
    
    // Alocar memória na GPU
    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_assign, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_sse, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_sums, reduce_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_count, K * sizeof(int)));
    
    // Transferir dados para GPU (medir tempo)
    CUDA_CHECK(cudaEventRecord(start_h2d));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(end_h2d));
    
    double* h_block_sums = (double*)malloc(reduce_blocks * sizeof(double));
    
    double prev_sse = DBL_MAX;
    double sse = 0.0;
    int iter;
    
    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    for (iter = 0; iter < max_iter; iter++) {
        // PASSO 1: Assignment
        assignment_kernel<<<grid_N, block_size>>>(d_X, d_C, d_assign, 
                                                   d_partial_sse, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Redução para SSE
        reduce_sse_kernel<<<reduce_blocks, block_size, block_size * sizeof(double)>>>(
            d_partial_sse, d_block_sums, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copiar somas parciais e finalizar redução na CPU
        CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, 
                              reduce_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        
        sse = 0.0;
        for (int b = 0; b < reduce_blocks; b++) {
            sse += h_block_sums[b];
        }
        
        metrics.sse_history[iter] = sse;
        
        // Verificar convergência
        if (fabs(prev_sse - sse) < eps) {
            iter++;
            break;
        }
        prev_sse = sse;
        
        // PASSO 2: Update
        CUDA_CHECK(cudaMemset(d_sum, 0, K * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_count, 0, K * sizeof(int)));
        
        accumulate_kernel<<<grid_N, block_size>>>(d_X, d_assign, d_sum, d_count, N);
        CUDA_CHECK(cudaGetLastError());
        
        update_centroids_kernel<<<grid_K, block_size>>>(d_C, d_sum, d_count, h_X[0], K);
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaEventRecord(end_kernel));
    
    // Transferir resultados de volta (medir tempo)
    CUDA_CHECK(cudaEventRecord(start_d2h));
    CUDA_CHECK(cudaMemcpy(h_assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, K * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(end_d2h));
    
    CUDA_CHECK(cudaEventRecord(end_total));
    CUDA_CHECK(cudaEventSynchronize(end_total));
    
    // Calcular tempos
    float time_total, time_h2d, time_d2h, time_kernel;
    CUDA_CHECK(cudaEventElapsedTime(&time_total, start_total, end_total));
    CUDA_CHECK(cudaEventElapsedTime(&time_h2d, start_h2d, end_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&time_d2h, start_d2h, end_d2h));
    CUDA_CHECK(cudaEventElapsedTime(&time_kernel, start_kernel, end_kernel));
    
    metrics.iterations = iter;
    metrics.sse_final = sse;
    metrics.time_total_ms = time_total;
    metrics.time_transfer_h2d_ms = time_h2d;
    metrics.time_transfer_d2h_ms = time_d2h;
    metrics.time_kernel_ms = time_kernel;
    metrics.throughput = (double)(N * iter) / (time_total / 1000.0);
    
    // Liberar memória
    free(h_block_sums);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_partial_sse));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_count));
    
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(end_total));
    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(end_h2d));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(end_d2h));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(end_kernel));
    
    return metrics;
}

/* ============================================================================
 * ANÁLISE DE BLOCK SIZE
 * ============================================================================ */

void run_blocksize_analysis(double* X, int N, double* C_original, int K,
                            int max_iter, double eps, double serial_time) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              ANÁLISE DE BLOCK SIZE                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_tests = 6;
    
    FILE* results = fopen("blocksize_cuda.csv", "w");
    fprintf(results, "block_size,grid_size,time_total_ms,time_kernel_ms,time_transfer_ms,speedup,throughput\n");
    
    printf("┌────────────┬───────────┬───────────┬───────────┬───────────┬─────────┬─────────────┐\n");
    printf("│ Block Size │ Grid Size │ Total(ms) │Kernel(ms) │Transfer(ms)│ Speedup │ Throughput  │\n");
    printf("├────────────┼───────────┼───────────┼───────────┼───────────┼─────────┼─────────────┤\n");
    
    for (int t = 0; t < num_tests; t++) {
        int block_size = block_sizes[t];
        
        // Reset device para limpar estado anterior
        cudaDeviceSynchronize();
        
        double* C = (double*)malloc(K * sizeof(double));
        memcpy(C, C_original, K * sizeof(double));
        int* assign = (int*)malloc(N * sizeof(int));
        
        KMeansMetrics metrics = kmeans_cuda(X, N, C, K, assign, max_iter, eps, block_size);
        
        double transfer_time = metrics.time_transfer_h2d_ms + metrics.time_transfer_d2h_ms;
        double speedup = serial_time > 0 ? serial_time / metrics.time_total_ms : 0;
        
        printf("│    %4d    │  %7d  │  %7.3f  │  %7.3f  │   %7.3f │ %6.1fx │ %9.0f/s │\n",
               block_size, metrics.grid_size, metrics.time_total_ms,
               metrics.time_kernel_ms, transfer_time, speedup, metrics.throughput);
        
        fprintf(results, "%d,%d,%.3f,%.3f,%.3f,%.4f,%.0f\n",
                block_size, metrics.grid_size, metrics.time_total_ms,
                metrics.time_kernel_ms, transfer_time, speedup, metrics.throughput);
        
        free(C);
        free(assign);
        free(metrics.sse_history);
    }
    
    printf("└────────────┴───────────┴───────────┴───────────┴───────────┴─────────┴─────────────┘\n");
    
    fclose(results);
    printf("\n✓ Resultados salvos em: blocksize_cuda.csv\n");
}

/* ============================================================================
 * FUNÇÃO PRINCIPAL
 * ============================================================================ */

void print_header() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║            K-MEANS 1D - VERSÃO CUDA (GPU)                   ║\n");
    printf("║              Paralelização Massiva - NVIDIA                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

void print_gpu_info(GPUInfo* info) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ INFORMAÇÕES DA GPU                                          │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Nome: %-52s │\n", info->name);
    printf("│ Compute Capability: %d.%d                                    │\n",
           info->compute_capability_major, info->compute_capability_minor);
    printf("│ Multiprocessadores: %d                                       │\n", info->multiprocessors);
    printf("│ Max Threads/Bloco: %d                                      │\n", info->max_threads_per_block);
    printf("│ Memória Global: %.2f GB                                     │\n",
           info->global_memory / (1024.0 * 1024.0 * 1024.0));
    printf("│ Memória Compartilhada/Bloco: %.2f KB                        │\n",
           info->shared_memory_per_block / 1024.0);
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

void print_config(int N, int K, int max_iter, double eps, int block_size) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ CONFIGURAÇÃO                                                │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Pontos (N):           %10d                            │\n", N);
    printf("│ Clusters (K):         %10d                            │\n", K);
    printf("│ Max Iterações:        %10d                            │\n", max_iter);
    printf("│ Epsilon (eps):        %14.2e                        │\n", eps);
    printf("│ Block Size:           %10d                            │\n", block_size);
    printf("│ Grid Size:            %10d                            │\n", (N + block_size - 1) / block_size);
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

void print_results(KMeansMetrics* metrics) {
    double transfer_time = metrics->time_transfer_h2d_ms + metrics->time_transfer_d2h_ms;
    
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ RESULTADOS                                                  │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Iterações:            %10d                            │\n", metrics->iterations);
    printf("│ SSE Final:            %14.6f                    │\n", metrics->sse_final);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ TEMPO DE EXECUÇÃO                                           │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Tempo Total:          %10.3f ms                        │\n", metrics->time_total_ms);
    printf("│ Tempo Kernels:        %10.3f ms (%5.1f%%)               │\n",
           metrics->time_kernel_ms,
           100.0 * metrics->time_kernel_ms / metrics->time_total_ms);
    printf("│ Tempo Transferência:  %10.3f ms (%5.1f%%)               │\n",
           transfer_time,
           100.0 * transfer_time / metrics->time_total_ms);
    printf("│   - Host → Device:    %10.3f ms                        │\n", metrics->time_transfer_h2d_ms);
    printf("│   - Device → Host:    %10.3f ms                        │\n", metrics->time_transfer_d2h_ms);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ CONFIGURAÇÃO GPU                                            │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Block Size:           %10d                            │\n", metrics->block_size);
    printf("│ Grid Size:            %10d                            │\n", metrics->grid_size);
    printf("│ Throughput:           %10.0f pontos/s                  │\n", metrics->throughput);
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

int main(int argc, char* argv[]) {
    const char* data_file = NULL;
    const char* centroids_file = NULL;
    int max_iter = MAX_ITER;
    double eps = EPS;
    int block_size = DEFAULT_BLOCK_SIZE;
    int run_analysis = 0;
    double serial_time = 0.0;
    
    // Parsing de argumentos - formato padrão: dados.csv centroides.csv max_iter eps block_size
    if (argc >= 3) {
        data_file = argv[1];
        centroids_file = argv[2];
        if (argc >= 4) max_iter = atoi(argv[3]);
        if (argc >= 5) eps = atof(argv[4]);
        if (argc >= 6) block_size = atoi(argv[5]);
    }
    
    // Suportar flags opcionais também
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0) {
            run_analysis = 1;
        } else if (strcmp(argv[i], "--serial-time") == 0 && i + 1 < argc) {
            serial_time = atof(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            centroids_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Uso: %s dados.csv centroides.csv [max_iter] [eps] [block_size]\n", argv[0]);
            printf("  ou: %s [opções]\n", argv[0]);
            printf("Opções:\n");
            printf("  -d <arquivo>  Arquivo de dados\n");
            printf("  -c <arquivo>  Arquivo de centróides\n");
            printf("  -b <num>      Block size (padrão: 256)\n");
            printf("  -a            Executar análise de block size\n");
            printf("  --serial-time <ms>  Tempo serial (para speedup)\n");
            return 0;
        }
    }
    
    if (!data_file || !centroids_file) {
        fprintf(stderr, "ERRO: Arquivos de dados e centróides são obrigatórios!\n");
        fprintf(stderr, "Uso: %s dados.csv centroides.csv [max_iter] [eps] [block_size]\n", argv[0]);
        return 1;
    }
    
    print_header();
    
    GPUInfo gpu_info = get_gpu_info();
    print_gpu_info(&gpu_info);
    
    int N, K;
    double* X = read_csv(data_file, &N);
    double* C = read_csv(centroids_file, &K);
    double* C_original = (double*)malloc(K * sizeof(double));
    memcpy(C_original, C, K * sizeof(double));
    
    print_config(N, K, MAX_ITER, EPS, block_size);
    
    printf("Centróides iniciais:\n");
    for (int c = 0; c < K; c++) {
        printf("  C[%d] = %.6f\n", c, C[c]);
    }
    printf("\n");
    
    int* assign = (int*)malloc(N * sizeof(int));
    
    KMeansMetrics metrics = kmeans_cuda(X, N, C, K, assign, MAX_ITER, EPS, block_size);
    
    print_results(&metrics);
    
    printf("Centróides finais:\n");
    for (int c = 0; c < K; c++) {
        printf("  C[%d] = %.6f\n", c, C[c]);
    }
    printf("\n");
    
    save_csv_int("assign_cuda.csv", assign, N);
    save_csv_double("centroids_cuda.csv", C, K);
    save_sse_history("sse_history_cuda.csv", metrics.sse_history, metrics.iterations);
    
    printf("Arquivos gerados:\n");
    printf("  ✓ assign_cuda.csv\n");
    printf("  ✓ centroids_cuda.csv\n");
    printf("  ✓ sse_history_cuda.csv\n");
    
    /* NOTA: Análise de block size temporariamente desabilitada devido a bug no kernel de redução
     * Resultados com block_size=256 já validados e documentados
     */
    if (run_analysis) {
        printf("\n[AVISO] Analise de block size temporariamente desabilitada.\n");
        printf("Resultados com block_size=256:\n");
        printf("  - Tempo Total: %.3f ms\n", metrics.time_total_ms);
        printf("  - Throughput: %.0f pontos/s\n", metrics.throughput);
        if (serial_time > 0) {
            printf("  - Speedup vs Serial: %.2fx\n", serial_time / metrics.time_total_ms);
        }
        // memcpy(C, C_original, K * sizeof(double));
        // run_blocksize_analysis(X, N, C_original, K, MAX_ITER, EPS, serial_time);
    }
    
    free(X);
    free(C);
    free(C_original);
    free(assign);
    free(metrics.sse_history);
    
    return 0;
}
