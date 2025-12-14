# Benchmark Completo - Serial vs OpenMP
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK: K-means 1D - Dataset Pequeno" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

cd ..

Write-Host "=== SERIAL (Baseline) ===" -ForegroundColor Yellow
Write-Host "Executando..."
$output = & .\serial\kmeans_1d_naive.exe data\dados_pequeno.csv data\dados_pequeno_centroides_init.csv 50 1e-6
$output | Select-String "Iterações|Tempo|SSE final"

Write-Host "`n=== OPENMP (Paralelo) ===" -ForegroundColor Yellow

$threads = @(1, 2, 4, 8)
foreach ($t in $threads) {
    Write-Host "`nThreads: $t" -ForegroundColor Green
    $output = & .\openmp\kmeans_1d_parallel.exe data\dados_pequeno.csv data\dados_pequeno_centroides_init.csv 50 1e-6 $t static 0
    $output | Select-String "Threads:|Iterações|Tempo|SSE final"
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Benchmark concluído!" -ForegroundColor Green
Write-Host "================================================`n" -ForegroundColor Cyan
