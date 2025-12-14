# Script para gerar todos os datasets
Write-Host "Gerando datasets..." -ForegroundColor Cyan

cd ..\data

# Dataset médio
Write-Host "`nDataset MÉDIO (100K pontos, 8 clusters)..." -ForegroundColor Yellow
python generate_data.py --N 100000 --K 8 --output dados_medio --seed 43

# Dataset grande  
Write-Host "`nDataset GRANDE (1M pontos, 16 clusters)..." -ForegroundColor Yellow
python generate_data.py --N 1000000 --K 16 --output dados_grande --seed 44

Write-Host "`nTodos os datasets gerados!" -ForegroundColor Green
