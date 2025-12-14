/* kmeans_1d_omp_cuda.cu
   K-means 1D híbrido: OpenMP + CUDA
   - Divide dados em chunks entre threads OpenMP
   - Cada thread OpenMP coordena operações CUDA de forma assíncrona
   - Usa cudaStreamCreate para overlapping de operações

   Compilar: nvcc -O2 -Xcompiler -fopenmp kmeans_1d_omp_cuda.cu -o kmeans_1d_omp_cuda -lm
   Uso:      ./kmeans_1d_omp_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [threads=2] [block_size=256] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>


static int count_rows(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        exit(1);
    }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws)
            rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out)
{
    int R = count_rows(path);
    if (R <= 0)
    {
        fprintf(stderr, "Arquivo vazio: %s\n", path);
        exit(1);
    }
    double *A = (double *)malloc((size_t)R * sizeof(double));
    if (!A)
    {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        exit(1);
    }

    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        exit(1);
    }

    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (only_ws)
            continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok)
        {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            fclose(f);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r > R)
            break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- CUDA kernel ---------- */
__global__ void kernel_assignment(const double *X, const double *C, int *assign, 
                                   double *sse_per_point, int N, int K, int offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    sse_per_point[i] = 0.0;

    int best = 0;
    double bestd = 1e300;

    for (int c = 0; c < K; c++)
    {
        double diff = X[i] - C[c];
        double d = diff * diff;
        if (d < bestd)
        {
            bestd = d;
            best = c;
        }
    }

    assign[i] = best;
    sse_per_point[i] = bestd;
}

/* ---------- Híbrido OpenMP + CUDA ---------- */
static void kmeans_1d_hybrid_omp_cuda(const double *X_host, double *C_host, int *assign_host,
                                       int N, int K, int max_iter, double eps, 
                                       int num_threads, int block_size,
                                       int *iters_out, double *sse_out)
{
    omp_set_num_threads(num_threads);
    
    typedef struct {
        cudaStream_t stream;
        double *X_dev;
        double *C_dev;
        double *sse_dev;
        int *assign_dev;
        double *sse_local;
        int start;
        int local_N;
    } thread_context_t;

    thread_context_t *contexts = (thread_context_t *)malloc(num_threads * sizeof(thread_context_t));

    /* Initialize thread contexts */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* Divisão de trabalho entre threads OpenMP */
        int chunk_size = (N + nthreads - 1) / nthreads;
        int start = tid * chunk_size;
        int end = start + chunk_size;
        if (end > N) end = N;
        int local_N = end - start;

        contexts[tid].start = start;
        contexts[tid].local_N = local_N;

        if (local_N > 0)
        {
            cudaStreamCreate(&contexts[tid].stream);
            cudaMalloc(&contexts[tid].X_dev, local_N * sizeof(double));
            cudaMalloc(&contexts[tid].C_dev, K * sizeof(double));
            cudaMalloc(&contexts[tid].assign_dev, local_N * sizeof(int));
            cudaMalloc(&contexts[tid].sse_dev, local_N * sizeof(double));
            
            double *sse_zero = (double *)calloc(local_N, sizeof(double));
            cudaMemcpyAsync(contexts[tid].sse_dev, sse_zero, local_N * sizeof(double), 
                           cudaMemcpyHostToDevice, contexts[tid].stream);
            cudaStreamSynchronize(contexts[tid].stream);
            free(sse_zero);
            
            contexts[tid].sse_local = (double *)malloc(local_N * sizeof(double));

            cudaMemcpyAsync(contexts[tid].X_dev, &X_host[start], local_N * sizeof(double), 
                           cudaMemcpyHostToDevice, contexts[tid].stream);
            cudaStreamSynchronize(contexts[tid].stream);
        }
    }
    
    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it;

    printf("\n--- SSE por iteração (OpenMP+CUDA) ---\n");

    for (it = 0; it < max_iter; it++)
    {
        sse_global = 0.0;

        /* Assignment: cada thread OpenMP processa um chunk via CUDA */
        #pragma omp parallel reduction(+:sse_global)
        {
            int tid = omp_get_thread_num();
            int local_N = contexts[tid].local_N;
            int start = contexts[tid].start;

            if (local_N > 0)
            {
                cudaMemcpyAsync(contexts[tid].C_dev, C_host, K * sizeof(double), 
                               cudaMemcpyHostToDevice, contexts[tid].stream);
                cudaStreamSynchronize(contexts[tid].stream);

                cudaMemsetAsync(contexts[tid].sse_dev, 0, local_N * sizeof(double), contexts[tid].stream);
                cudaStreamSynchronize(contexts[tid].stream);

                /* Kernel assignment */
                int grid_size = (local_N + block_size - 1) / block_size;
                kernel_assignment<<<grid_size, block_size, 0, contexts[tid].stream>>>(
                    contexts[tid].X_dev, contexts[tid].C_dev, contexts[tid].assign_dev, 
                    contexts[tid].sse_dev, local_N, K, start);

                cudaStreamSynchronize(contexts[tid].stream);

                /* Copia resultados de volta (assíncrono) */
                cudaMemcpyAsync(&assign_host[start], contexts[tid].assign_dev, local_N * sizeof(int), 
                               cudaMemcpyDeviceToHost, contexts[tid].stream);
                cudaMemcpyAsync(contexts[tid].sse_local, contexts[tid].sse_dev, local_N * sizeof(double), 
                               cudaMemcpyDeviceToHost, contexts[tid].stream);

                /* Sincroniza stream */
                cudaStreamSynchronize(contexts[tid].stream);

                /* Redução local do SSE */
                double sse_thread = 0.0;
                for (int i = 0; i < local_N; i++)
                    sse_thread += contexts[tid].sse_local[i];
                sse_global += sse_thread;
            }
        }

        printf("Iteração %d: SSE = %.6f\n", it + 1, sse_global);

        /* Convergência */
        if (it > 0) {
            double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
            if (rel < eps)
            {
                printf("Convergiu (variação relativa < %.6f)\n", eps);
                it++;
                break;
            }
        }

        /* Update no host (sequencial) */
        double *sum = (double *)calloc((size_t)K, sizeof(double));
        int *cnt = (int *)calloc((size_t)K, sizeof(int));

        for (int i = 0; i < N; i++)
        {
            int a = assign_host[i];
            cnt[a]++;
            sum[a] += X_host[i];
        }

        for (int c = 0; c < K; c++)
        {
            if (cnt[c] > 0)
                C_host[c] = sum[c] / (double)cnt[c];
            else
                C_host[c] = X_host[0];
        }

        free(sum);
        free(cnt);
        prev_sse = sse_global;
    }

    printf("--------------------------------------\n\n");

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (contexts[tid].local_N > 0)
        {
            free(contexts[tid].sse_local);
            cudaFree(contexts[tid].X_dev);
            cudaFree(contexts[tid].C_dev);
            cudaFree(contexts[tid].assign_dev);
            cudaFree(contexts[tid].sse_dev);
            cudaStreamDestroy(contexts[tid].stream);
        }
    }

    free(contexts);

    *iters_out = it;
    *sse_out = sse_global;
}

/* ---------- main ---------- */
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [threads=2] [block_size=256] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: Híbrido OpenMP + CUDA - divide dados entre threads OpenMP, cada uma usando CUDA\n");
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    int num_threads = (argc > 5) ? atoi(argv[5]) : 2;
    int block_size = (argc > 6) ? atoi(argv[6]) : 256;

    const char *outAssign = (argc > 7) ? argv[7] : NULL;
    const char *outCentroid = (argc > 8) ? argv[8] : NULL;

    if (max_iter <= 0 || eps <= 0.0 || num_threads <= 0)
    {
        fprintf(stderr, "Parâmetros inválidos\n");
        return 1;
    }

    int N = 0, K = 0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int *)malloc((size_t)N * sizeof(int));

    if (!assign)
    {
        fprintf(stderr, "Sem memoria para assign\n");
        free(X);
        free(C);
        return 1;
    }

    double t0 = omp_get_wtime();
    int iters = 0;
    double sse = 0.0;

    kmeans_1d_hybrid_omp_cuda(X, C, assign, N, K, max_iter, eps, 
                               num_threads, block_size, &iters, &sse);

    double t1 = omp_get_wtime();
    double ms = 1000.0 * (t1 - t0);

    printf("=== K-means 1D (Híbrido: OpenMP + CUDA) ===\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Threads OpenMP: %d | Block size CUDA: %d\n", num_threads, block_size);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);
    printf("============================================\n");

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign);
    free(X);
    free(C);
    return 0;
}
