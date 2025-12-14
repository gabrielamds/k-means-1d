# Benchmark COMPLETO - Serial, OpenMP, MPI e Hybrid
Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK COMPLETO: K-means 1D - TODAS IMPLEMENTAÇÕES" -ForegroundColor Cyan
Write-Host "=======================================================`n" -ForegroundColor Cyan

cd ..

$datasets = @(
    @{name="pequeno"; N="10K"; K=4},
    @{name="medio"; N="100K"; K=8},
    @{name="grande"; N="1M"; K=16}
)

foreach ($ds in $datasets) {
    $dsname = $ds.name
    $N = $ds.N
    $K = $ds.K
    
    Write-Host "`n=======================================================" -ForegroundColor Yellow
    Write-Host "  Dataset: $($dsname.ToUpper()) (N=$N, K=$K)" -ForegroundColor Yellow
    Write-Host "=======================================================" -ForegroundColor Yellow
    
    # Serial
    Write-Host "`n[SERIAL - Baseline]" -ForegroundColor Magenta
    $output = & .\serial\kmeans_1d_naive.exe "data\dados_$dsname.csv" "data\dados_${dsname}_centroides_init.csv" 50 1e-6 2>&1
    $time_match = $output | Select-String "Tempo:.*?(\d+\.\d+)\s*ms"
    $time_serial = if ($time_match) { $time_match.Matches.Groups[1].Value } else { "N/A" }
    Write-Host "  Tempo: $time_serial ms" -ForegroundColor White
    
    # OpenMP
    Write-Host "`n[OPENMP - Shared Memory]" -ForegroundColor Green
    foreach ($t in @(2, 4, 8)) {
        $output = & .\openmp\kmeans_1d_parallel.exe "data\dados_$dsname.csv" "data\dados_${dsname}_centroides_init.csv" 50 1e-6 $t static 0 2>&1
        $time_match = $output | Select-String "Tempo:.*?(\d+\.\d+)\s*ms"
        if ($time_match) {
            $time_omp = $time_match.Matches.Groups[1].Value
            if ($time_serial -ne "N/A") {
                $speedup = [math]::Round([double]$time_serial / [double]$time_omp, 2)
                Write-Host "  $t threads: $time_omp ms (Speedup: ${speedup}x)" -ForegroundColor White
            } else {
                Write-Host "  $t threads: $time_omp ms" -ForegroundColor White
            }
        }
    }
    
    # MPI
    Write-Host "`n[MPI - Distributed Memory]" -ForegroundColor Cyan
    foreach ($p in @(2, 4, 8)) {
        $output = mpiexec -n $p .\mpi\kmeans_1d_mpi.exe "data\dados_$dsname.csv" "data\dados_${dsname}_centroides_init.csv" 50 1e-6 2>&1
        $time_match = $output | Select-String "Tempo total:.*?(\d+\.\d+)\s*ms"
        if ($time_match) {
            $time_mpi = $time_match.Matches.Groups[1].Value
            if ($time_serial -ne "N/A") {
                $speedup = [math]::Round([double]$time_serial / [double]$time_mpi, 2)
                Write-Host "  $p processos: $time_mpi ms (Speedup: ${speedup}x)" -ForegroundColor White
            } else {
                Write-Host "  $p processos: $time_mpi ms" -ForegroundColor White
            }
        }
    }
    
    # Hybrid
    Write-Host "`n[HYBRID - OpenMP + MPI]" -ForegroundColor Magenta
    $configs = @(@{p=2;t=2}, @{p=2;t=4}, @{p=4;t=2})
    foreach ($cfg in $configs) {
        $env:OMP_NUM_THREADS = $cfg.t
        $output = mpiexec -n $cfg.p .\hybrid\kmeans_1d_omp_mpi.exe "data\dados_$dsname.csv" "data\dados_${dsname}_centroides_init.csv" 50 1e-6 static 0 2>&1
        $time_match = $output | Select-String "Tempo total:.*?(\d+\.\d+)\s*ms"
        if ($time_match) {
            $time_hybrid = $time_match.Matches.Groups[1].Value
            $total = $cfg.p * $cfg.t
            if ($time_serial -ne "N/A") {
                $speedup = [math]::Round([double]$time_serial / [double]$time_hybrid, 2)
                Write-Host "  $($cfg.p)p x $($cfg.t)t ($total total): $time_hybrid ms (Speedup: ${speedup}x)" -ForegroundColor White
            } else {
                Write-Host "  $($cfg.p)p x $($cfg.t)t ($total total): $time_hybrid ms" -ForegroundColor White
            }
        }
    }
}

Write-Host "`n=======================================================`n" -ForegroundColor Cyan
Write-Host "  Benchmark COMPLETO concluído!" -ForegroundColor Green
Write-Host "=======================================================`n" -ForegroundColor Cyan
