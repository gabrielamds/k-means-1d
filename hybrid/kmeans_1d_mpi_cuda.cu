/* kmeans_1d_mpi_cuda.cu
   K-means 1D híbrido: MPI + CUDA
   - Cada processo MPI controla uma GPU
   - MPI distribui dados entre nós/GPUs
   - CUDA acelera assignment dentro de cada processo
   - MPI_Allreduce sincroniza centróides entre processos

   Compilar: mpicc -O2 kmeans_1d_mpi_cuda.cu -o kmeans_1d_mpi_cuda -lm -L/usr/local/cuda/lib64 -lcudart
           ou: nvcc -O2 -arch=sm_70 -Xcompiler -fopenmp kmeans_1d_mpi_cuda.cu -o kmeans_1d_mpi_cuda -lmpi
   Uso:      mpirun -np 2 ./kmeans_1d_mpi_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [block_size=256] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>


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

/* ---------- k-means híbrido: MPI + CUDA ---------- */
static void kmeans_1d_mpi_cuda(const double *X_local, double *C, int *assign_local,
                                int N_local, int K, int max_iter, double eps, 
                                int block_size, int *iters_out, double *sse_out, int rank)
{
    /* Cada processo MPI usa uma GPU diferente (se disponível) */
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    int device_id = rank % num_devices;
    cudaSetDevice(device_id);

    /* Alocação GPU para partição local */
    double *X_dev, *C_dev, *sse_dev;
    int *assign_dev;

    cudaMalloc(&X_dev, N_local * sizeof(double));
    cudaMalloc(&C_dev, K * sizeof(double));
    cudaMalloc(&assign_dev, N_local * sizeof(int));
    cudaMalloc(&sse_dev, N_local * sizeof(double));

    double *sse_per_point = (double *)malloc(N_local * sizeof(double));

    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it;

    if (rank == 0)
        printf("\n--- SSE por iteração (MPI+CUDA) ---\n");

    for (it = 0; it < max_iter; it++)
    {
        /* Copia dados para GPU */
        cudaMemcpy(X_dev, X_local, N_local * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(C_dev, C, K * sizeof(double), cudaMemcpyHostToDevice);

        /* Kernel assignment na GPU local */
        int grid_size = (N_local + block_size - 1) / block_size;
        kernel_assignment<<<grid_size, block_size>>>(X_dev, C_dev, assign_dev, sse_dev, N_local, K);

        /* Copia resultados de volta */
        cudaMemcpy(assign_local, assign_dev, N_local * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(sse_per_point, sse_dev, N_local * sizeof(double), cudaMemcpyDeviceToHost);

        /* Redução SSE local */
        double sse_local = 0.0;
        for (int i = 0; i < N_local; i++)
            sse_local += sse_per_point[i];

        /* Redução SSE global via MPI */
        MPI_Allreduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0)
            printf("Iteração %d: SSE = %.6f\n", it + 1, sse_global);

        /* Convergência */
        double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps)
        {
            if (rank == 0)
                printf("Convergiu (variação relativa < %.6f)\n", eps);
            it++;
            break;
        }

        /* Update: acumulação local no host */
        double *sum_local = (double *)calloc((size_t)K, sizeof(double));
        int *cnt_local = (int *)calloc((size_t)K, sizeof(int));

        for (int i = 0; i < N_local; i++)
        {
            int a = assign_local[i];
            cnt_local[a]++;
            sum_local[a] += X_local[i];
        }

        /* Redução global via MPI */
        double *sum_global = (double *)calloc((size_t)K, sizeof(double));
        int *cnt_global = (int *)calloc((size_t)K, sizeof(int));

        MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        /* Calcula novos centróides */
        for (int c = 0; c < K; c++)
        {
            if (cnt_global[c] > 0)
                C[c] = sum_global[c] / (double)cnt_global[c];
            else
                C[c] = 0.0;
        }

        free(sum_local);
        free(cnt_local);
        free(sum_global);
        free(cnt_global);

        prev_sse = sse_global;
    }

    if (rank == 0)
        printf("-----------------------------------\n\n");

    *iters_out = it;
    *sse_out = sse_global;

    /* Limpeza GPU */
    cudaFree(X_dev);
    cudaFree(C_dev);
    cudaFree(assign_dev);
    cudaFree(sse_dev);
    free(sse_per_point);
}

/* ---------- main ---------- */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3)
    {
        if (rank == 0)
        {
            printf("Uso: mpirun -np P %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [block_size=256] [assign.csv] [centroids.csv]\n", argv[0]);
            printf("Obs: Híbrido MPI + CUDA - cada processo MPI controla uma GPU\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    int block_size = (argc > 5) ? atoi(argv[5]) : 256;
    const char *outAssign = (argc > 6) ? argv[6] : NULL;
    const char *outCentroid = (argc > 7) ? argv[7] : NULL;

    int N = 0, K = 0;
    double *X_full = NULL;
    double *C = NULL;

    if (rank == 0)
    {
        X_full = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        C = (double *)malloc((size_t)K * sizeof(double));

    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int N_local = N / size;
    int remainder = N % size;
    if (rank < remainder)
        N_local++;

    double *X_local = (double *)malloc((size_t)N_local * sizeof(double));
    int *assign_local = (int *)malloc((size_t)N_local * sizeof(int));

    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        sendcounts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));
        int offset = 0;
        for (int p = 0; p < size; p++)
        {
            int count = N / size;
            if (p < remainder)
                count++;
            sendcounts[p] = count;
            displs[p] = offset;
            offset += count;
        }
    }

    MPI_Scatterv(X_full, sendcounts, displs, MPI_DOUBLE,
                 X_local, N_local, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    int iters = 0;
    double sse = 0.0;
    kmeans_1d_mpi_cuda(X_local, C, assign_local, N_local, K, max_iter, eps, 
                       block_size, &iters, &sse, rank);
    double t1 = MPI_Wtime();
    double ms = 1000.0 * (t1 - t0);

    int *assign_full = NULL;
    if (rank == 0)
        assign_full = (int *)malloc((size_t)N * sizeof(int));

    MPI_Gatherv(assign_local, N_local, MPI_INT,
                assign_full, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int num_devices;
        cudaGetDeviceCount(&num_devices);

        printf("=== K-means 1D (Híbrido: MPI + CUDA) ===\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Processos MPI: %d | GPUs disponíveis: %d | Block size: %d\n", 
               size, num_devices, block_size);
        printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);
        printf("========================================\n");

        write_assign_csv(outAssign, assign_full, N);
        write_centroids_csv(outCentroid, C, K);

        free(X_full);
        free(assign_full);
        free(sendcounts);
        free(displs);
    }

    free(X_local);
    free(assign_local);
    free(C);

    MPI_Finalize();
    return 0;
}
