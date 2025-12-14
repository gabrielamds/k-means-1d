# Benchmark Serial
Write-Host "=== Benchmark Serial (Baseline) ===" -ForegroundColor Cyan
Write-Host ""

$SERIAL_BIN = "..\serial\kmeans_1d_naive.exe"
$MAX_ITER = 50
$EPS = "1e-6"

Write-Host "Dataset: PEQUENO (N=10k, K=4)" -ForegroundColor Yellow
& $SERIAL_BIN "..\data\dados_pequeno.csv" "..\data\dados_pequeno_centroides_init.csv" $MAX_ITER $EPS

Write-Host "`n---`n"

Write-Host "Dataset: MÉDIO (N=100k, K=8)" -ForegroundColor Yellow
& $SERIAL_BIN "..\data\dados_medio.csv" "..\data\dados_medio_centroides_init.csv" $MAX_ITER $EPS

Write-Host "`n---`n"

Write-Host "Dataset: GRANDE (N=1M, K=16)" -ForegroundColor Yellow
& $SERIAL_BIN "..\data\dados_grande.csv" "..\data\dados_grande_centroides_init.csv" $MAX_ITER $EPS

Write-Host "`n==================================="
