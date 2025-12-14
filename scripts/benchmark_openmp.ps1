# Benchmark OpenMP
Write-Host "=== Benchmark OpenMP ===" -ForegroundColor Cyan
Write-Host ""

$OMP_BIN = "..\openmp\kmeans_1d_parallel.exe"
$MAX_ITER = 50
$EPS = "1e-6"
$THREADS = @(1, 2, 4, 8)

foreach ($DATA in @("pequeno", "medio", "grande")) {
    if ($DATA -eq "pequeno") {
        $N = "10k"
        $K = 4
    } elseif ($DATA -eq "medio") {
        $N = "100k"
        $K = 8
    } else {
        $N = "1M"
        $K = 16
    }
    
    Write-Host "Dataset: $($DATA.ToUpper()) (N=$N, K=$K)" -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($T in $THREADS) {
        Write-Host "  Threads: $T | Schedule: static" -ForegroundColor Gray
        & $OMP_BIN "..\data\dados_$DATA.csv" "..\data\dados_${DATA}_centroides_init.csv" `
                   $MAX_ITER $EPS $T "static" 0
        Write-Host ""
    }
    
    Write-Host "---`n"
}

Write-Host "==================================="
