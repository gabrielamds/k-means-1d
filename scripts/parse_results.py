#!/usr/bin/env python3
"""
Script para parsear resultados de benchmarks e extrair métricas
"""

import re
import sys
from pathlib import Path

def parse_serial_results(text):
    """Parseia resultados da versão serial"""
    results = []
    
    # Regex patterns
    pattern_n = r'N=(\d+)'
    pattern_k = r'K=(\d+)'
    pattern_iters = r'Iterações:\s*(\d+)'
    pattern_sse = r'SSE final:\s*([\d.]+)'
    pattern_time = r'Tempo:\s*([\d.]+)\s*ms'
    
    sections = text.split('Dataset:')
    
    for section in sections[1:]:  # Pula primeira seção vazia
        dataset_name = section.split('\n')[0].strip()
        
        n_match = re.search(pattern_n, section)
        k_match = re.search(pattern_k, section)
        iters_match = re.search(pattern_iters, section)
        sse_match = re.search(pattern_sse, section)
        time_match = re.search(pattern_time, section)
        
        if all([n_match, k_match, time_match]):
            results.append({
                'dataset': dataset_name,
                'N': int(n_match.group(1)),
                'K': int(k_match.group(1)),
                'iterations': int(iters_match.group(1)) if iters_match else None,
                'sse': float(sse_match.group(1)) if sse_match else None,
                'time_ms': float(time_match.group(1)),
                'version': 'serial'
            })
    
    return results

def parse_openmp_results(text):
    """Parseia resultados da versão OpenMP"""
    results = []
    
    pattern_threads = r'Threads:\s*(\d+)'
    pattern_schedule = r'Schedule:\s*(\w+)'
    pattern_time = r'Tempo:\s*([\d.]+)\s*ms'
    
    current_dataset = None
    
    for line in text.split('\n'):
        if 'Dataset:' in line:
            current_dataset = line.split('Dataset:')[1].strip()
        elif 'Threads:' in line:
            threads_match = re.search(pattern_threads, line)
            schedule_match = re.search(pattern_schedule, line)
            
            # Busca próxima linha com tempo
            if threads_match:
                threads = int(threads_match.group(1))
                schedule = schedule_match.group(1) if schedule_match else 'static'
                # Armazena para próxima linha
                # (simplificado - numa implementação real, seria mais robusto)
    
    return results

def parse_cuda_results(text):
    """Parseia resultados da versão CUDA"""
    results = []
    
    pattern_block = r'Block size:\s*(\d+)'
    pattern_h2d = r'H2D.*:\s*([\d.]+)\s*ms'
    pattern_kernel = r'Kernel.*:\s*([\d.]+)\s*ms'
    pattern_d2h = r'D2H.*:\s*([\d.]+)\s*ms'
    pattern_total = r'Tempo total.*:\s*([\d.]+)\s*ms'
    pattern_throughput = r'Throughput:\s*([\d.]+)\s*Mpontos/s'
    
    # Similar ao serial, parseia seções
    return results

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 parse_results.py <arquivo_resultados>")
        sys.exit(1)
    
    result_file = Path(sys.argv[1])
    
    if not result_file.exists():
        print(f"Erro: arquivo {result_file} não encontrado")
        sys.exit(1)
    
    text = result_file.read_text()
    
    # Detecta tipo de resultado pelo nome do arquivo
    if 'serial' in result_file.name:
        results = parse_serial_results(text)
    elif 'openmp' in result_file.name:
        results = parse_openmp_results(text)
    elif 'cuda' in result_file.name:
        results = parse_cuda_results(text)
    else:
        print(f"Tipo de resultado não reconhecido: {result_file.name}")
        sys.exit(1)
    
    # Imprime resultados parseados
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
