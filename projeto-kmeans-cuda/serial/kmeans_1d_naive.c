// kmeans_1d_naive.c - Versão Sequencial
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ==================== FUNÇÕES AUXILIARES CSV ====================

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

// ==================== ALGORITMO K-MEANS ====================

double assignment_step_1d(const double *X, const double *C, int *assign, int N, int K) {
    double sse = 0.0;
    for(int i=0; i<N; i++) {
        int best = -1;
        double bestd = 1e300;
        for(int c=0; c<K; c++) {
            double diff = X[i] - C[c];
            double d = diff * diff;
            if(d < bestd) {
                bestd = d;
                best = c;
            }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

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
            C[c] = X[0]; // cluster vazio
    }
    
    free(sum);
    free(cnt);
}

void kmeans_1d(const double *X, double *C, int *assign, int N, int K, 
               int max_iter, double eps) {
    double prev_sse = 1e300;
    
    printf("\nIteração | SSE\n");
    printf("---------|---------------\n");
    
    for(int it=0; it<max_iter; it++) {
        // Assignment
        double sse = assignment_step_1d(X, C, assign, N, K);
        
        printf("%8d | %.6lf\n", it, sse);
        
        // Verificar convergência
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps) {
            printf("\nConvergiu após %d iterações!\n", it+1);
            break;
        }
        
        // Update
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
}

// ==================== MAIN ====================

int main(int argc, char **argv) {
    if(argc < 7) {
        printf("Uso: %s dados.csv centroides.csv max_iter eps assign.csv centroids.csv\n", argv[0]);
        return 1;
    }
    
    const char *dados_file = argv[1];
    const char *centroides_file = argv[2];
    int max_iter = atoi(argv[3]);
    double eps = atof(argv[4]);
    const char *assign_out = argv[5];
    const char *centroids_out = argv[6];
    
    // Carregar dados
    int N, K;
    double *X = read_csv_1col(dados_file, &N);
    double *C = read_csv_1col(centroides_file, &K);
    int *assign = (int*)malloc(N * sizeof(int));
    
    printf("=== K-MEANS 1D SEQUENCIAL ===\n");
    printf("N = %d pontos, K = %d clusters\n", N, K);
    
    // Executar K-means
    clock_t inicio = clock();
    kmeans_1d(X, C, assign, N, K, max_iter, eps);
    clock_t fim = clock();
    
    double tempo_ms = ((double)(fim - inicio) / CLOCKS_PER_SEC) * 1000.0;
    printf("\nTempo de execução: %.2lf ms\n", tempo_ms);
    
    // Salvar resultados
    write_assign_csv(assign_out, assign, N);
    write_centroids_csv(centroids_out, C, K);
    
    printf("\nResultados salvos em %s e %s\n", assign_out, centroids_out);
    
    // Liberar memória
    free(X);
    free(C);
    free(assign);
    
    return 0;
}
