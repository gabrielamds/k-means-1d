# Benchmark MPI
Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK MPI" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

cd ..

$datasets = @(
    @{name="pequeno"; N="10K"; K=4},
    @{name="medio"; N="100K"; K=8},
    @{name="grande"; N="1M"; K=16}
)

$procs = @(1, 2, 4, 8)

foreach ($ds in $datasets) {
    $dsname = $ds.name
    $N = $ds.N
    $K = $ds.K
    
    Write-Host "`n============================================" -ForegroundColor Yellow
    Write-Host "  Dataset: $($dsname.ToUpper()) (N=$N, K=$K)" -ForegroundColor Yellow
    Write-Host "============================================" -ForegroundColor Yellow
    
    Write-Host "`n[MPI - Distributed]" -ForegroundColor Green
    foreach ($p in $procs) {
        $output = mpiexec -n $p .\mpi\kmeans_1d_mpi.exe "data\dados_$dsname.csv" "data\dados_${dsname}_centroides_init.csv" 50 1e-6 2>&1
        $time_match = $output | Select-String "Tempo:.*?(\d+\.\d+)\s*ms"
        if ($time_match) {
            $time_mpi = $time_match.Matches.Groups[1].Value
            Write-Host "  $p processos: $time_mpi ms" -ForegroundColor White
        }
    }
}

Write-Host "`n=========================================`n" -ForegroundColor Cyan
Write-Host "  Benchmark MPI concluído!" -ForegroundColor Green
Write-Host "=========================================`n" -ForegroundColor Cyan
