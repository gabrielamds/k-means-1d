# Benchmark Hybrid (OpenMP + MPI)
Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK HYBRID (OpenMP + MPI)" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

cd ..

$datasets = @(
    @{name="pequeno"; N="10K"; K=4},
    @{name="medio"; N="100K"; K=8},
    @{name="grande"; N="1M"; K=16}
)

$configs = @(
    @{procs=2; threads=2},
    @{procs=2; threads=4},
    @{procs=4; threads=2}
)

foreach ($ds in $datasets) {
    $dsname = $ds.name
    $N = $ds.N
    $K = $ds.K
    
    Write-Host "`n============================================" -ForegroundColor Yellow
    Write-Host "  Dataset: $($dsname.ToUpper()) (N=$N, K=$K)" -ForegroundColor Yellow
    Write-Host "============================================" -ForegroundColor Yellow
    
    Write-Host "`n[HYBRID - OpenMP + MPI]" -ForegroundColor Magenta
    foreach ($cfg in $configs) {
        $p = $cfg.procs
        $t = $cfg.threads
        $env:OMP_NUM_THREADS = $t
        $output = mpiexec -n $p .\hybrid\kmeans_1d_omp_mpi.exe "data\dados_$dsname.csv" "data\dados_${dsname}_centroides_init.csv" 50 1e-6 static 0 2>&1
        $time_match = $output | Select-String "Tempo:.*?(\d+\.\d+)\s*ms"
        if ($time_match) {
            $time_hybrid = $time_match.Matches.Groups[1].Value
            Write-Host "  $p procs x $t threads: $time_hybrid ms" -ForegroundColor White
        }
    }
}

Write-Host "`n=========================================`n" -ForegroundColor Cyan
Write-Host "  Benchmark Hybrid concluído!" -ForegroundColor Green
Write-Host "=========================================`n" -ForegroundColor Cyan
