#!/usr/bin/env python3
"""
Análise de speedup: compara todas as versões vs baseline serial
Gera gráficos de speedup vs # recursos (threads, processos, etc.)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_results():
    """Carrega resultados dos arquivos (simplificado)"""
    # Na implementação real, parsearia os arquivos de resultados
    # Por ora, retorna dados de exemplo
    
    results = {
        'serial': {
            'pequeno': {'time': 100},
            'medio': {'time': 1000},
            'grande': {'time': 10000}
        },
        'openmp': {
            'pequeno': {1: 100, 2: 52, 4: 28, 8: 16, 16: 12},
            'medio': {1: 1000, 2: 520, 4: 280, 8: 160, 16: 120},
            'grande': {1: 10000, 2: 5200, 4: 2800, 8: 1600, 16: 1200}
        },
        'cuda': {
            'pequeno': {128: 15, 256: 12, 512: 14},
            'medio': {128: 95, 256: 85, 512: 90},
            'grande': {128: 850, 256: 800, 512: 820}
        },
        'mpi': {
            'pequeno': {1: 100, 2: 55, 4: 32, 8: 22},
            'medio': {1: 1000, 2: 550, 4: 320, 8: 220},
            'grande': {1: 10000, 2: 5500, 4: 3200, 8: 2200}
        }
    }
    
    return results

def calculate_speedup(baseline_time, parallel_times):
    """Calcula speedup: T_serial / T_parallel"""
    speedups = {}
    for config, time in parallel_times.items():
        speedups[config] = baseline_time / time
    return speedups

def plot_speedup(results, dataset, output_dir):
    """Gera gráfico de speedup para um dataset"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline = results['serial'][dataset]['time']
    
    # OpenMP
    if dataset in results['openmp']:
        threads = sorted(results['openmp'][dataset].keys())
        speedups = [baseline / results['openmp'][dataset][t] for t in threads]
        ax.plot(threads, speedups, 'o-', label='OpenMP', linewidth=2, markersize=8)
    
    # MPI
    if dataset in results['mpi']:
        procs = sorted(results['mpi'][dataset].keys())
        speedups = [baseline / results['mpi'][dataset][p] for p in procs]
        ax.plot(procs, speedups, 's-', label='MPI', linewidth=2, markersize=8)
    
    # CUDA (speedup médio dos block sizes)
    if dataset in results['cuda']:
        cuda_times = list(results['cuda'][dataset].values())
        avg_cuda_time = np.mean(cuda_times)
        cuda_speedup = baseline / avg_cuda_time
        ax.axhline(y=cuda_speedup, color='g', linestyle='--', label=f'CUDA (avg)', linewidth=2)
    
    # Linha ideal (speedup = # recursos)
    max_x = max(threads) if 'openmp' in results and dataset in results['openmp'] else 16
    ax.plot([1, max_x], [1, max_x], 'k:', label='Ideal (linear)', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Número de Recursos (threads/processos)', fontsize=12)
    ax.set_ylabel('Speedup vs Serial', fontsize=12)
    ax.set_title(f'Speedup - Dataset {dataset.upper()}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(0, max_x + 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'speedup_{dataset}.png', dpi=150)
    plt.close()

def plot_speedup_comparison(results, output_dir):
    """Gera gráfico comparativo de speedup para todos os datasets"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = ['pequeno', 'medio', 'grande']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        baseline = results['serial'][dataset]['time']
        
        # OpenMP
        if dataset in results['openmp']:
            threads = sorted(results['openmp'][dataset].keys())
            speedups = [baseline / results['openmp'][dataset][t] for t in threads]
            ax.plot(threads, speedups, 'o-', label='OpenMP', linewidth=2)
        
        # MPI
        if dataset in results['mpi']:
            procs = sorted(results['mpi'][dataset].keys())
            speedups = [baseline / results['mpi'][dataset][p] for p in procs]
            ax.plot(procs, speedups, 's-', label='MPI', linewidth=2)
        
        # Linha ideal
        max_x = 16
        ax.plot([1, max_x], [1, max_x], 'k:', label='Ideal', alpha=0.5)
        
        ax.set_xlabel('# Recursos')
        ax.set_ylabel('Speedup')
        ax.set_title(f'{dataset.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_x + 1)
    
    plt.suptitle('Comparação de Speedup por Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=150)
    plt.close()

def main():
    # Setup
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / 'report' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analisando speedup...")
    
    # Carrega resultados
    results = load_results()
    
    # Gera gráficos individuais
    for dataset in ['pequeno', 'medio', 'grande']:
        print(f"  Gerando gráfico para dataset {dataset}...")
        plot_speedup(results, dataset, output_dir)
    
    # Gera gráfico comparativo
    print("  Gerando gráfico comparativo...")
    plot_speedup_comparison(results, output_dir)
    
    print(f"\nGráficos salvos em: {output_dir}")
    print("  - speedup_pequeno.png")
    print("  - speedup_medio.png")
    print("  - speedup_grande.png")
    print("  - speedup_comparison.png")

if __name__ == "__main__":
    main()
