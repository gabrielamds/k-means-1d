/* kmeans_1d_parallel.c
   K-means 1D (C99) com paralelização OpenMP:
   - Assignment: paralelizado com reduction(+:sse)
   - Update: paralelizado com acumuladores por thread (Option A)

   Compilar: gcc -O2 -std=c99 -fopenmp kmeans_1d_parallel.c -o kmeans_1d_parallel -lm
   Uso:      ./kmeans_1d_parallel dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [threads=0] [schedule=static] [chunk=0] [assign.csv] [centroids.csv]
             threads=0 usa todos os cores disponíveis
             schedule: static, dynamic, guided
             chunk=0 usa chunk size padrão
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static int g_chunk_size = 0;
static omp_sched_t g_schedule_type = omp_sched_static;

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

/* ---------- k-means 1D com OpenMP ---------- */

/* Paralelizado o loop principal com reduction para SSE */
static double assignment_step_1d(const double *X, const double *C, int *assign, int N, int K)
{
    double sse = 0.0;

#pragma omp parallel for reduction(+ : sse) schedule(runtime)
    for (int i = 0; i < N; i++)
    {
        int best = -1;
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
        sse += bestd;
    }
    return sse;
}

/* Paralelizado com acumuladores por thread (Option A) */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K)
{
    int num_threads = omp_get_max_threads();

    /* Aloca acumuladores por thread: sum_thread[thread_id][cluster_id] */
    double **sum_thread = (double **)malloc((size_t)num_threads * sizeof(double *));
    int **cnt_thread = (int **)malloc((size_t)num_threads * sizeof(int *));

    if (!sum_thread || !cnt_thread)
    {
        fprintf(stderr, "Sem memoria no update\n");
        exit(1);
    }

    for (int t = 0; t < num_threads; t++)
    {
        sum_thread[t] = (double *)calloc((size_t)K, sizeof(double));
        cnt_thread[t] = (int *)calloc((size_t)K, sizeof(int));
        if (!sum_thread[t] || !cnt_thread[t])
        {
            fprintf(stderr, "Sem memoria para acumuladores\n");
            exit(1);
        }
    }

/* Fase 1: Acumulação paralela (cada thread acumula em seu próprio array) */
#pragma omp parallel
    {
        int tid = omp_get_thread_num();

#pragma omp for schedule(runtime)
        for (int i = 0; i < N; i++)
        {
            int a = assign[i];
            cnt_thread[tid][a] += 1;
            sum_thread[tid][a] += X[i];
        }
    }

    /* Fase 2: Redução sequencial dos acumuladores e cálculo dos centróides */
    for (int c = 0; c < K; c++)
    {
        double total_sum = 0.0;
        int total_cnt = 0;

        for (int t = 0; t < num_threads; t++)
        {
            total_sum += sum_thread[t][c];
            total_cnt += cnt_thread[t][c];
        }

        if (total_cnt > 0)
            C[c] = total_sum / (double)total_cnt;
        else
            C[c] = X[0]; /* cluster vazio recebe o primeiro ponto */
    }

    /* Libera memória dos acumuladores */
    for (int t = 0; t < num_threads; t++)
    {
        free(sum_thread[t]);
        free(cnt_thread[t]);
    }
    free(sum_thread);
    free(cnt_thread);
}

static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    printf("\n--- SSE por iteração ---\n");
    for (it = 0; it < max_iter; it++)
    {
        sse = assignment_step_1d(X, C, assign, N, K);
        printf("Iteração %d: SSE = %.6f\n", it + 1, sse);

        /* parada por variação relativa do SSE */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps)
        {
            printf("Convergiu (variação relativa < %.6f)\n", eps);
            it++;
            break;
        }
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
    printf("------------------------\n\n");
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [threads=0] [schedule=static] [chunk=0] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        printf("     threads=0 usa todos os cores disponíveis\n");
        printf("     schedule: static, dynamic, guided\n");
        printf("     chunk=0 usa chunk size padrão\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;

    int num_threads = (argc > 5) ? atoi(argv[5]) : 0;
    const char *schedule_str = (argc > 6) ? argv[6] : "static";
    int chunk_size = (argc > 7) ? atoi(argv[7]) : 0;

    const char *outAssign = (argc > 8) ? argv[8] : NULL;
    const char *outCentroid = (argc > 9) ? argv[9] : NULL;

    if (max_iter <= 0 || eps <= 0.0)
    {
        fprintf(stderr, "Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    if (num_threads > 0)
    {
        omp_set_num_threads(num_threads);
    }

    if (strcmp(schedule_str, "dynamic") == 0)
    {
        g_schedule_type = omp_sched_dynamic;
    }
    else if (strcmp(schedule_str, "guided") == 0)
    {
        g_schedule_type = omp_sched_guided;
    }
    else
    {
        g_schedule_type = omp_sched_static;
    }

    g_chunk_size = chunk_size;
    omp_set_schedule(g_schedule_type, g_chunk_size);

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
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    double t1 = omp_get_wtime();
    double ms = 1000.0 * (t1 - t0);

    printf("=== K-means 1D (OpenMP paralelo) ===\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Threads: %d | Schedule: %s", omp_get_max_threads(), schedule_str);
    if (chunk_size > 0)
        printf(" (chunk=%d)", chunk_size);
    printf("\n");
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);
    printf("====================================\n");

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign);
    free(X);
    free(C);
    return 0;
}
