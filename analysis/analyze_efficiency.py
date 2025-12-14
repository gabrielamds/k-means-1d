#!/usr/bin/env python3
"""
Análise de eficiência paralela: Speedup / # recursos
Identifica ponto onde eficiência cai abaixo de threshold aceitável
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def calculate_efficiency(speedup, num_resources):
    """Eficiência = Speedup / # recursos"""
    return speedup / num_resources

def plot_efficiency(results, dataset, output_dir):
    """Gera gráfico de eficiência para um dataset"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline = results['serial'][dataset]['time']
    
    # OpenMP
    if dataset in results['openmp']:
        threads = sorted(results['openmp'][dataset].keys())
        speedups = [baseline / results['openmp'][dataset][t] for t in threads]
        efficiencies = [s / t * 100 for s, t in zip(speedups, threads)]  # Em %
        ax.plot(threads, efficiencies, 'o-', label='OpenMP', linewidth=2, markersize=8)
    
    # MPI
    if dataset in results['mpi']:
        procs = sorted(results['mpi'][dataset].keys())
        speedups = [baseline / results['mpi'][dataset][p] for p in procs]
        efficiencies = [s / p * 100 for s, p in zip(speedups, procs)]
        ax.plot(procs, efficiencies, 's-', label='MPI', linewidth=2, markersize=8)
    
    # Linhas de referência
    ax.axhline(y=100, color='g', linestyle='--', label='Ideal (100%)', alpha=0.7)
    ax.axhline(y=80, color='orange', linestyle=':', label='Threshold (80%)', alpha=0.7)
    ax.axhline(y=50, color='r', linestyle=':', label='Baixa (50%)', alpha=0.7)
    
    ax.set_xlabel('Número de Recursos', fontsize=12)
    ax.set_ylabel('Eficiência Paralela (%)', fontsize=12)
    ax.set_title(f'Eficiência Paralela - Dataset {dataset.upper()}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'efficiency_{dataset}.png', dpi=150)
    plt.close()

def plot_efficiency_comparison(results, output_dir):
    """Gera gráfico comparativo de eficiência"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dataset = 'grande'  # Usa dataset grande para comparação
    baseline = results['serial'][dataset]['time']
    
    # OpenMP
    if dataset in results['openmp']:
        threads = sorted(results['openmp'][dataset].keys())
        speedups = [baseline / results['openmp'][dataset][t] for t in threads]
        efficiencies = [s / t * 100 for s, t in zip(speedups, threads)]
        ax.plot(threads, efficiencies, 'o-', label='OpenMP', linewidth=2, markersize=8)
    
    # MPI
    if dataset in results['mpi']:
        procs = sorted(results['mpi'][dataset].keys())
        speedups = [baseline / results['mpi'][dataset][p] for p in procs]
        efficiencies = [s / p * 100 for s, p in zip(speedups, procs)]
        ax.plot(procs, efficiencies, 's-', label='MPI', linewidth=2, markersize=8)
    
    # Referências
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=80, color='orange', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Número de Recursos', fontsize=12)
    ax.set_ylabel('Eficiência (%)', fontsize=12)
    ax.set_title('Comparação de Eficiência (Dataset Grande)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison.png', dpi=150)
    plt.close()

def generate_efficiency_report(results):
    """Gera relatório textual sobre eficiência"""
    report = []
    report.append("# Análise de Eficiência\n")
    report.append("## Pontos de Diminishing Returns\n")
    
    for dataset in ['pequeno', 'medio', 'grande']:
        report.append(f"\n### Dataset: {dataset.upper()}\n")
        baseline = results['serial'][dataset]['time']
        
        # OpenMP
        if dataset in results['openmp']:
            report.append("**OpenMP:**\n")
            threads = sorted(results['openmp'][dataset].keys())
            for t in threads:
                speedup = baseline / results['openmp'][dataset][t]
                eff = speedup / t * 100
                status = "✓" if eff >= 80 else "⚠" if eff >= 50 else "✗"
                report.append(f"- {t} threads: {eff:.1f}% {status}\n")
        
        # MPI
        if dataset in results['mpi']:
            report.append("\n**MPI:**\n")
            procs = sorted(results['mpi'][dataset].keys())
            for p in procs:
                speedup = baseline / results['mpi'][dataset][p]
                eff = speedup / p * 100
                status = "✓" if eff >= 80 else "⚠" if eff >= 50 else "✗"
                report.append(f"- {p} processos: {eff:.1f}% {status}\n")
    
    return ''.join(report)

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / 'report' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analisando eficiência...")
    
    # Carrega resultados
    from analyze_speedup import load_results
    results = load_results()
    
    # Gera gráficos
    for dataset in ['pequeno', 'medio', 'grande']:
        print(f"  Gerando gráfico para dataset {dataset}...")
        plot_efficiency(results, dataset, output_dir)
    
    print("  Gerando gráfico comparativo...")
    plot_efficiency_comparison(results, output_dir)
    
    # Gera relatório textual
    print("  Gerando relatório de eficiência...")
    report = generate_efficiency_report(results)
    report_file = project_root / 'report' / 'EFFICIENCY_ANALYSIS.md'
    report_file.write_text(report)
    
    print(f"\nGráficos salvos em: {output_dir}")
    print("  - efficiency_pequeno.png")
    print("  - efficiency_medio.png")
    print("  - efficiency_grande.png")
    print("  - efficiency_comparison.png")
    print(f"\nRelatório salvo em: {report_file}")

if __name__ == "__main__":
    main()
