# Benchmark Completo - Todos os Datasets
Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK COMPLETO: K-means 1D" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

cd ..

$datasets = @(
    @{name="pequeno"; N="10K"; K=4},
    @{name="medio"; N="100K"; K=8},
    @{name="grande"; N="1M"; K=16}
)

$threads = @(1, 2, 4, 8)

foreach ($ds in $datasets) {
    $dsname = $ds.name
    $N = $ds.N
    $K = $ds.K
    
    Write-Host "`n============================================" -ForegroundColor Yellow
    Write-Host "  Dataset: $($dsname.ToUpper()) (N=$N, K=$K)" -ForegroundColor Yellow
    Write-Host "============================================" -ForegroundColor Yellow
    
    # Serial
    Write-Host "`n[SERIAL - Baseline]" -ForegroundColor Magenta
    $output = & .\serial\kmeans_1d_naive.exe "data\dados_$dsname.csv" "data\dados_${dsname}_centroides_init.csv" 50 1e-6
    $time_serial = ($output | Select-String "Tempo:.*?(\d+\.\d+)\s*ms").Matches.Groups[1].Value
    Write-Host "  Tempo: $time_serial ms" -ForegroundColor White
    
    # OpenMP
    Write-Host "`n[OPENMP - Paralelo]" -ForegroundColor Green
    foreach ($t in $threads) {
        $output = & .\openmp\kmeans_1d_parallel.exe "data\dados_$dsname.csv" "data\dados_${dsname}_centroides_init.csv" 50 1e-6 $t static 0
        $time_omp = ($output | Select-String "Tempo:.*?(\d+\.\d+)\s*ms").Matches.Groups[1].Value
        $speedup = [math]::Round([double]$time_serial / [double]$time_omp, 2)
        Write-Host "  $t threads: $time_omp ms (Speedup: ${speedup}x)" -ForegroundColor White
    }
}

Write-Host "`n=========================================`n" -ForegroundColor Cyan
Write-Host "  Benchmark concluído!" -ForegroundColor Green
Write-Host "=========================================`n" -ForegroundColor Cyan
