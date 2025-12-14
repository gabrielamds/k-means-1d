#!/usr/bin/env pwsh
# Script para executar benchmarks no Windows e gerar resultados_windows.csv

# Forçar cultura en-US para usar ponto como separador decimal
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  BENCHMARKS K-MEANS 1D - WINDOWS (CPU)    " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Criar diretório de resultados se não existir
New-Item -ItemType Directory -Force -Path "results" | Out-Null

# Array para armazenar resultados
$resultados = @()

# Configurações dos datasets
$datasets = @(
    @{Nome="pequeno"; N="10000"; K=4},
    @{Nome="medio"; N="100000"; K=8},
    @{Nome="grande"; N="1000000"; K=16}
)

# ============================================
# 1. SERIAL
# ============================================
Write-Host "[1/3] Executando Serial..." -ForegroundColor Yellow

foreach ($ds in $datasets) {
    $nome = $ds.Nome
    Write-Host "  Dataset: $($nome.ToUpper())" -NoNewline
    
    $output = & "serial\kmeans_1d_naive.exe" `
        "data\dados_$nome.csv" `
        "data\dados_${nome}_centroides_init.csv" `
        50 0.000001 2>&1 | Out-String
    
    # Extrair tempo (aceita "Tempo:" ou "Tempo total:")
    if ($output -match "Tempo(?:\s+total)?:\s*([\d.]+)\s*ms") {
        $tempo = [double]$Matches[1]
        Write-Host " -> $tempo ms" -ForegroundColor Green
    } else {
        $tempo = 0
        Write-Host " -> ERRO (não encontrou tempo no output)" -ForegroundColor Red
        Write-Host $output
    }
    
    # Extrair SSE
    $sse = 0.0
    if ($output -match "SSE final:\s*([\d.]+)") {
        $sse = [double]$Matches[1]
    }
    
    $resultados += [PSCustomObject]@{
        dataset = $nome
        implementacao = "Serial"
        config = "-"
        tempo_ms = $tempo
        sse = $sse
    }
}

# ============================================
# 2. OPENMP
# ============================================
Write-Host "`n[2/3] Executando OpenMP..." -ForegroundColor Yellow

$threads = @(2, 4, 8)

foreach ($ds in $datasets) {
    $nome = $ds.Nome
    Write-Host "  Dataset: $($nome.ToUpper())"
    
    foreach ($t in $threads) {
        Write-Host "    $t threads" -NoNewline
        
        $env:OMP_NUM_THREADS = $t
        $output = & "openmp\kmeans_1d_parallel.exe" `
            "data\dados_$nome.csv" `
            "data\dados_${nome}_centroides_init.csv" `
            50 0.000001 2>&1 | Out-String
        
        # Extrair tempo (aceita "Tempo:" ou "Tempo total:")
        if ($output -match "Tempo(?:\s+total)?:\s*([\d.]+)\s*ms") {
            $tempo = [double]$Matches[1]
            Write-Host " -> $tempo ms" -ForegroundColor Green
        } else {
            $tempo = 0
            Write-Host " -> ERRO" -ForegroundColor Red
        }
        
        # Extrair SSE
        $sse = 0.0
        if ($output -match "SSE final:\s*([\d.]+)") {
            $sse = [double]$Matches[1]
        }
        
        $resultados += [PSCustomObject]@{
            dataset = $nome
            implementacao = "OpenMP"
            config = "${t}t"
            tempo_ms = $tempo
            sse = $sse
        }
    }
}

# ============================================
# 3. MPI
# ============================================
Write-Host "`n[3/3] Executando MPI..." -ForegroundColor Yellow

$processos = @(2)

foreach ($ds in $datasets) {
    $nome = $ds.Nome
    Write-Host "  Dataset: $($nome.ToUpper())"
    
    foreach ($p in $processos) {
        Write-Host "    $p processos" -NoNewline
        
        $output = & "mpiexec" -n $p "mpi\kmeans_1d_mpi.exe" `
            "data\dados_$nome.csv" `
            "data\dados_${nome}_centroides_init.csv" `
            50 0.000001 2>&1 | Out-String
        
        # Extrair tempo (aceita "Tempo:" ou "Tempo total:")
        if ($output -match "Tempo(?:\s+total)?:\s*([\d.]+)\s*ms") {
            $tempo = [double]$Matches[1]
            Write-Host " -> $tempo ms" -ForegroundColor Green
        } else {
            $tempo = 0
            Write-Host " -> ERRO" -ForegroundColor Red
        }
        
        # Extrair SSE
        $sse = 0.0
        if ($output -match "SSE final:\s*([\d.]+)") {
            $sse = [double]$Matches[1]
        }
        
        $resultados += [PSCustomObject]@{
            dataset = $nome
            implementacao = "MPI"
            config = "${p}p"
            tempo_ms = $tempo
            sse = $sse
        }
    }
}

# ============================================
# SALVAR RESULTADOS
# ============================================
Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "SALVANDO RESULTADOS" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

$csvPath = "results\resultados_windows.csv"
$resultados | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "✓ Resultados salvos em: $csvPath" -ForegroundColor Green
Write-Host "✓ Total de medições: $($resultados.Count)" -ForegroundColor Green

# Mostrar resumo
Write-Host "`nResumo por implementação:" -ForegroundColor Yellow
$resultados | Group-Object implementacao | ForEach-Object {
    Write-Host "  $($_.Name): $($_.Count) medições"
}

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "Agora você pode usar estes resultados no Colab!" -ForegroundColor Green
Write-Host "Execute: !python scripts/analise_completa.py" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
