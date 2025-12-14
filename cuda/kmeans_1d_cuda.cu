/* kmeans_1d_cuda.cu
   K-means 1D com CUDA (GPU):
   - Assignment: kernel parallelizado (1 thread por ponto)
   - Update: Opção A - copiar assign para CPU, calcular médias no host

   Compilar: nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda -lm
   Uso:      ./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [block_size=256] [assign.csv] [centroids.csv]
             block_size: 128, 256, 512
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
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

/* ---------- CUDA kernels ---------- */
/* Kernel de assignment: cada thread processa 1 ponto */
__global__ void kernel_assignment(const double *X, const double *C, int *assign, 
                                   double *sse_per_point, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

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

/* ---------- CPU helper functions ---------- */
static double reduce_sse_host(const double *sse_per_point, int N)
{
    double sse = 0.0;
    for (int i = 0; i < N; i++)
        sse += sse_per_point[i];
    return sse;
}

static void update_step_host(const double *X, double *C, const int *assign, 
                              int N, int K)
{
    double *sum = (double *)calloc((size_t)K, sizeof(double));
    int *cnt = (int *)calloc((size_t)K, sizeof(int));

    if (!sum || !cnt)
    {
        fprintf(stderr, "Sem memoria no update\n");
        exit(1);
    }

    for (int i = 0; i < N; i++)
    {
        int a = assign[i];
        cnt[a]++;
        sum[a] += X[i];
    }

    for (int c = 0; c < K; c++)
    {
        if (cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = X[0];
    }

    free(sum);
    free(cnt);
}

/* ---------- K-means 1D com CUDA ---------- */
static void kmeans_1d_cuda(const double *X_host, double *C_host, int *assign_host,
                           int N, int K, int max_iter, double eps, int block_size,
                           int *iters_out, double *sse_out,
                           double *time_h2d_out, double *time_kernel_out, 
                           double *time_d2h_out)
{
    /* Alocação de memória na GPU */
    double *X_dev, *C_dev, *sse_dev;
    int *assign_dev;

    cudaMalloc(&X_dev, N * sizeof(double));
    cudaMalloc(&C_dev, K * sizeof(double));
    cudaMalloc(&assign_dev, N * sizeof(int));
    cudaMalloc(&sse_dev, N * sizeof(double));

    double *sse_per_point = (double *)malloc(N * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total_h2d = 0.0, total_kernel = 0.0, total_d2h = 0.0;

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;

    printf("\n--- SSE por iteração ---\n");

    for (it = 0; it < max_iter; it++)
    {
        /* H2D: X */
        cudaEventRecord(start);
        cudaMemcpy(X_dev, X_host, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_h2d_x;
        cudaEventElapsedTime(&ms_h2d_x, start, stop);
        total_h2d += ms_h2d_x;

        /* H2D: C */
        cudaEventRecord(start);
        cudaMemcpy(C_dev, C_host, K * sizeof(double), cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_h2d_c;
        cudaEventElapsedTime(&ms_h2d_c, start, stop);
        total_h2d += ms_h2d_c;

        /* Kernel: assignment */
        int grid_size = (N + block_size - 1) / block_size;
        cudaEventRecord(start);
        kernel_assignment<<<grid_size, block_size>>>(X_dev, C_dev, assign_dev, sse_dev, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_kernel;
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_kernel += ms_kernel;

        /* D2H: assign e sse */
        cudaEventRecord(start);
        cudaMemcpy(assign_host, assign_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(sse_per_point, sse_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_d2h;
        cudaEventElapsedTime(&ms_d2h, start, stop);
        total_d2h += ms_d2h;

        /* Redução SSE no host */
        sse = reduce_sse_host(sse_per_point, N);
        printf("Iteração %d: SSE = %.6f\n", it + 1, sse);

        /* Verificar convergência */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps)
        {
            printf("Convergiu (variação relativa < %.6f)\n", eps);
            it++;
            break;
        }

        /* Update no host */
        update_step_host((const double *)X_host, C_host, (const int *)assign_host, N, K);
        prev_sse = sse;
    }

    printf("------------------------\n\n");

    *iters_out = it;
    *sse_out = sse;
    *time_h2d_out = total_h2d;
    *time_kernel_out = total_kernel;
    *time_d2h_out = total_d2h;

    /* Limpeza GPU */
    cudaFree(X_dev);
    cudaFree(C_dev);
    cudaFree(assign_dev);
    cudaFree(sse_dev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(sse_per_point);
}

/* ---------- main ---------- */
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [block_size=256] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        printf("     block_size: 128, 256, 512 (padrão 256)\n");
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    int block_size = (argc > 5) ? atoi(argv[5]) : 256;

    const char *outAssign = (argc > 6) ? argv[6] : NULL;
    const char *outCentroid = (argc > 7) ? argv[7] : NULL;

    if (max_iter <= 0 || eps <= 0.0)
    {
        fprintf(stderr, "Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    /* Validar block_size */
    if (block_size != 128 && block_size != 256 && block_size != 512)
    {
        fprintf(stderr, "Block size deve ser 128, 256 ou 512\n");
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

    cudaSetDevice(0);

    double t0 = (double)clock() / CLOCKS_PER_SEC;
    int iters = 0;
    double sse = 0.0;
    double time_h2d = 0.0, time_kernel = 0.0, time_d2h = 0.0;

    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, block_size,
                   &iters, &sse, &time_h2d, &time_kernel, &time_d2h);

    double t1 = (double)clock() / CLOCKS_PER_SEC;
    double ms_total = 1000.0 * (t1 - t0);

    printf("=== K-means 1D (CUDA - GPU) ===\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Block size: %d | Grid size: %d\n", block_size, (N + block_size - 1) / block_size);
    printf("Iterações: %d | SSE final: %.6f\n", iters, sse);
    printf("\nTempo por componente:\n");
    printf("  H2D (Host→Device): %.2f ms\n", time_h2d);
    printf("  Kernel (Assignment): %.2f ms\n", time_kernel);
    printf("  D2H (Device→Host): %.2f ms\n", time_d2h);
    printf("  Tempo total (wallclock): %.1f ms\n", ms_total);
    printf("  Throughput: %.2f Mpontos/s\n", (double)N * iters / (ms_total / 1000.0) / 1e6);
    printf("================================\n");

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign);
    free(X);
    free(C);
    return 0;
}
