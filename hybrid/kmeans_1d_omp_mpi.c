/* kmeans_1d_omp_mpi.c
   K-means 1D híbrido: OpenMP + MPI
   - MPI distribui dados entre nós (inter-node parallelism)
   - OpenMP paraleliza dentro de cada nó (intra-node parallelism)
   - Exemplo: 2 processos MPI × 4 threads OpenMP = 8 workers totais

   Compilar: mpicc -O2 -std=c99 -fopenmp kmeans_1d_omp_mpi.c -o kmeans_1d_omp_mpi -lm
   Uso:      mpirun -np 2 ./kmeans_1d_omp_mpi dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [threads=4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
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

/* ---------- k-means híbrido: MPI + OpenMP ---------- */

/* Assignment: paralelizado com OpenMP dentro de cada processo MPI */
static double assignment_step_hybrid(const double *X_local, const double *C, int *assign_local, int N_local, int K)
{
    double sse_local = 0.0;

    #pragma omp parallel for reduction(+:sse_local)
    for (int i = 0; i < N_local; i++)
    {
        int best = -1;
        double bestd = 1e300;
        for (int c = 0; c < K; c++)
        {
            double diff = X_local[i] - C[c];
            double d = diff * diff;
            if (d < bestd)
            {
                bestd = d;
                best = c;
            }
        }
        assign_local[i] = best;
        sse_local += bestd;
    }
    return sse_local;
}

/* Update: OpenMP para acumulação local, depois MPI_Allreduce */
static void update_step_hybrid(const double *X_local, double *C, const int *assign_local, int N_local, int K)
{
    int num_threads = omp_get_max_threads();
    
    /* Acumuladores por thread */
    double **sum_thread = (double **)malloc((size_t)num_threads * sizeof(double *));
    int **cnt_thread = (int **)malloc((size_t)num_threads * sizeof(int *));

    for (int t = 0; t < num_threads; t++)
    {
        sum_thread[t] = (double *)calloc((size_t)K, sizeof(double));
        cnt_thread[t] = (int *)calloc((size_t)K, sizeof(int));
    }

    /* Acumulação paralela com OpenMP */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < N_local; i++)
        {
            int a = assign_local[i];
            cnt_thread[tid][a] += 1;
            sum_thread[tid][a] += X_local[i];
        }
    }

    /* Redução local (sequencial dentro do processo) */
    double *sum_local = (double *)calloc((size_t)K, sizeof(double));
    int *cnt_local = (int *)calloc((size_t)K, sizeof(int));

    for (int c = 0; c < K; c++)
    {
        for (int t = 0; t < num_threads; t++)
        {
            sum_local[c] += sum_thread[t][c];
            cnt_local[c] += cnt_thread[t][c];
        }
    }

    /* Redução global entre processos MPI */
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

    /* Limpeza */
    for (int t = 0; t < num_threads; t++)
    {
        free(sum_thread[t]);
        free(cnt_thread[t]);
    }
    free(sum_thread);
    free(cnt_thread);
    free(sum_local);
    free(cnt_local);
    free(sum_global);
    free(cnt_global);
}

static void kmeans_1d_hybrid(const double *X_local, double *C, int *assign_local,
                              int N_local, int K, int max_iter, double eps,
                              int *iters_out, double *sse_out, int rank)
{
    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it;

    if (rank == 0)
        printf("\n--- SSE por iteração (OpenMP+MPI) ---\n");

    for (it = 0; it < max_iter; it++)
    {
        double sse_local = assignment_step_hybrid(X_local, C, assign_local, N_local, K);

        /* Redução global do SSE via MPI */
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

        update_step_hybrid(X_local, C, assign_local, N_local, K);
        prev_sse = sse_global;
    }

    if (rank == 0)
        printf("-------------------------------------\n\n");

    *iters_out = it;
    *sse_out = sse_global;
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
            printf("Uso: mpirun -np P %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [threads=4] [assign.csv] [centroids.csv]\n", argv[0]);
            printf("Obs: Híbrido OpenMP + MPI\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    int num_threads = (argc > 5) ? atoi(argv[5]) : 4;
    const char *outAssign = (argc > 6) ? argv[6] : NULL;
    const char *outCentroid = (argc > 7) ? argv[7] : NULL;

    if (num_threads > 0)
        omp_set_num_threads(num_threads);

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
    kmeans_1d_hybrid(X_local, C, assign_local, N_local, K, max_iter, eps, &iters, &sse, rank);
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
        printf("=== K-means 1D (Híbrido: OpenMP + MPI) ===\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Processos MPI: %d | Threads OpenMP: %d | Total workers: %d\n", 
               size, num_threads, size * num_threads);
        printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);
        printf("==========================================\n");

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
