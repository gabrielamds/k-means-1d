/* kmeans_1d_mpi.c
   K-means 1D (C99) com paralelização MPI (distributed memory):
   - Distribui N pontos entre P processos MPI
   - Assignment: cada processo calcula para sua partição local
   - Update: MPI_Allreduce para somar SSE, sums e counts globalmente
   - MPI_Bcast dos centróides atualizados no início de cada iteração

   Compilar: mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
   Uso:      mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ---------- util CSV 1D ---------- */
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

/* ---------- k-means 1D com MPI ---------- */

/* Assignment step: calcula apenas para a partição local */
static double assignment_step_1d_local(const double *X_local, const double *C, int *assign_local, int N_local, int K)
{
    double sse_local = 0.0;
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

/* Update step: acumula localmente e depois usa MPI_Allreduce */
static void update_step_1d_mpi(const double *X_local, double *C, const int *assign_local, int N_local, int K)
{
    double *sum_local = (double *)calloc((size_t)K, sizeof(double));
    int *cnt_local = (int *)calloc((size_t)K, sizeof(int));
    double *sum_global = (double *)calloc((size_t)K, sizeof(double));
    int *cnt_global = (int *)calloc((size_t)K, sizeof(int));

    if (!sum_local || !cnt_local || !sum_global || !cnt_global)
    {
        fprintf(stderr, "Sem memoria no update\n");
        exit(1);
    }

    /* Acumulação local */
    for (int i = 0; i < N_local; i++)
    {
        int a = assign_local[i];
        cnt_local[a] += 1;
        sum_local[a] += X_local[i];
    }

    /* Redução global: soma os acumuladores de todos os processos */
    MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /* Calcula novos centróides (todos os processos calculam) */
    for (int c = 0; c < K; c++)
    {
        if (cnt_global[c] > 0)
            C[c] = sum_global[c] / (double)cnt_global[c];
        else
            C[c] = 0.0; /* cluster vazio (raro em partições distribuídas) */
    }

    free(sum_local);
    free(cnt_local);
    free(sum_global);
    free(cnt_global);
}

static void kmeans_1d_mpi(const double *X_local, double *C, int *assign_local,
                          int N_local, int K, int max_iter, double eps,
                          int *iters_out, double *sse_out, int rank, double *comm_time)
{
    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it;
    double total_comm_time = 0.0;

    if (rank == 0)
        printf("\n--- SSE por iteração (MPI) ---\n");

    for (it = 0; it < max_iter; it++)
    {
        /* Assignment local */
        double sse_local = assignment_step_1d_local(X_local, C, assign_local, N_local, K);

        /* Redução global do SSE */
        double t_comm_start = MPI_Wtime();
        MPI_Allreduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        total_comm_time += (MPI_Wtime() - t_comm_start);

        if (rank == 0)
            printf("Iteração %d: SSE = %.6f\n", it + 1, sse_global);

        /* Critério de parada */
        double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps)
        {
            if (rank == 0)
                printf("Convergiu (variação relativa < %.6f)\n", eps);
            it++;
            break;
        }

        /* Update com MPI_Allreduce */
        double t_update_start = MPI_Wtime();
        update_step_1d_mpi(X_local, C, assign_local, N_local, K);
        total_comm_time += (MPI_Wtime() - t_update_start);

        prev_sse = sse_global;
    }

    if (rank == 0)
        printf("------------------------------\n\n");

    *iters_out = it;
    *sse_out = sse_global;
    *comm_time = total_comm_time;
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
            printf("Uso: mpirun -np P %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
            printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc > 5) ? argv[5] : NULL;
    const char *outCentroid = (argc > 6) ? argv[6] : NULL;

    if (max_iter <= 0 || eps <= 0.0)
    {
        if (rank == 0)
            fprintf(stderr, "Parâmetros inválidos: max_iter>0 e eps>0\n");
        MPI_Finalize();
        return 1;
    }

    /* Rank 0 lê os dados e centróides iniciais */
    int N = 0, K = 0;
    double *X_full = NULL;
    double *C = NULL;

    if (rank == 0)
    {
        X_full = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
    }

    /* Broadcast N e K para todos os processos */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Aloca centróides em todos os processos */
    if (rank != 0)
        C = (double *)malloc((size_t)K * sizeof(double));

    /* Broadcast dos centróides iniciais */
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Calcula partição local para cada processo */
    int N_local = N / size;
    int remainder = N % size;
    if (rank < remainder)
        N_local++;

    double *X_local = (double *)malloc((size_t)N_local * sizeof(double));
    int *assign_local = (int *)malloc((size_t)N_local * sizeof(int));

    if (!X_local || !assign_local)
    {
        fprintf(stderr, "Rank %d: Sem memoria para partição local\n", rank);
        MPI_Finalize();
        exit(1);
    }

    /* Distribui dados com MPI_Scatterv */
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

    /* Executa K-means distribuído */
    double t0 = MPI_Wtime();
    int iters = 0;
    double sse = 0.0;
    double comm_time = 0.0;
    kmeans_1d_mpi(X_local, C, assign_local, N_local, K, max_iter, eps, &iters, &sse, rank, &comm_time);
    double t1 = MPI_Wtime();
    double ms = 1000.0 * (t1 - t0);
    double comm_ms = 1000.0 * comm_time;

    /* Coleta assignments de volta ao rank 0 */
    int *assign_full = NULL;
    if (rank == 0)
        assign_full = (int *)malloc((size_t)N * sizeof(int));

    MPI_Gatherv(assign_local, N_local, MPI_INT,
                assign_full, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Rank 0 imprime resultados e salva arquivos */
    if (rank == 0)
    {
        printf("=== K-means 1D (MPI distribuído) ===\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Processos MPI: %d\n", size);
        printf("Iterações: %d | SSE final: %.6f\n", iters, sse);
        printf("Tempo total: %.1f ms\n", ms);
        printf("Tempo comunicação: %.1f ms (%.1f%%)\n", comm_ms, 100.0 * comm_ms / ms);
        printf("====================================\n");

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
