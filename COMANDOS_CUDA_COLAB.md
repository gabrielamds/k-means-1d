# Comandos para Executar CUDA no Google Colab

## 1️⃣ Setup Inicial (primeira célula)
```python
# Clonar repositório
!git clone https://github.com/gabrielamds/k-means-1d.git
%cd k-means-1d

# Verificar GPU
!nvidia-smi
```

## 2️⃣ Compilar CUDA (segunda célula)
```bash
%%bash
cd cuda
nvcc -O2 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda
cd ..
```

## 3️⃣ Gerar Dados (se necessário)
```bash
%%bash
cd data
# Dataset pequeno
python3 generate_data.py --N 10000 --K 4 --output dados_pequeno --seed 42
# Dataset médio
python3 generate_data.py --N 100000 --K 8 --output dados_medio --seed 43
# Dataset grande
python3 generate_data.py --N 1000000 --K 16 --output dados_grande --seed 44
cd ..
```

## 4️⃣ Executar CUDA - Dataset Pequeno
```bash
%%bash
./cuda/kmeans_1d_cuda data/dados_pequeno.csv data/dados_pequeno_centroides_init.csv 50 1e-6 256
```

## 5️⃣ Executar CUDA - Dataset Médio
```bash
%%bash
./cuda/kmeans_1d_cuda data/dados_medio.csv data/dados_medio_centroides_init.csv 50 1e-6 256
```

## 6️⃣ Executar CUDA - Dataset Grande
```bash
%%bash
./cuda/kmeans_1d_cuda data/dados_grande.csv data/dados_grande_centroides_init.csv 50 1e-6 256
```

## 7️⃣ Benchmark Completo (variando threads_per_block)
```bash
%%bash
echo "=== Benchmark CUDA - Dataset Grande ==="
echo ""
for TPB in 128 256 512 1024; do
    echo "Threads per block: $TPB"
    ./cuda/kmeans_1d_cuda data/dados_grande.csv data/dados_grande_centroides_init.csv 50 1e-6 $TPB | grep -E "Tempo|Iterações|SSE"
    echo ""
done
```

## 8️⃣ Compilar Hybrid (OpenMP + CUDA)
```bash
%%bash
cd hybrid
nvcc -O2 -arch=sm_75 -Xcompiler -fopenmp kmeans_1d_omp_cuda.cu -o kmeans_1d_omp_cuda
cd ..
```

## 9️⃣ Executar Hybrid (OpenMP + CUDA)
```bash
%%bash
# 2 threads OpenMP + GPU
OMP_NUM_THREADS=2 ./hybrid/kmeans_1d_omp_cuda data/dados_grande.csv data/dados_grande_centroides_init.csv 50 1e-6 256 static 0

# 4 threads OpenMP + GPU
OMP_NUM_THREADS=4 ./hybrid/kmeans_1d_omp_cuda data/dados_grande.csv data/dados_grande_centroides_init.csv 50 1e-6 256 static 0
```

## 🔟 Comparação Serial vs CUDA
```bash
%%bash
echo "=== Comparação Serial vs CUDA ==="
echo ""

# Compilar serial
cd serial
gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
cd ..

echo "[Serial]"
./serial/kmeans_1d_naive data/dados_grande.csv data/dados_grande_centroides_init.csv 50 1e-6 | grep -E "Tempo|Iterações"

echo ""
echo "[CUDA - 256 threads/block]"
./cuda/kmeans_1d_cuda data/dados_grande.csv data/dados_grande_centroides_init.csv 50 1e-6 256 | grep -E "Tempo|Iterações"
```

---

## 📝 Notas:
- `sm_75` é para Tesla T4 (GPU comum no Colab)
- Se der erro de arquitetura, use `sm_60` ou `sm_70`
- `threads_per_block` ideal: 256 ou 512
- Dataset grande (1M pontos) mostra melhor speedup da GPU
