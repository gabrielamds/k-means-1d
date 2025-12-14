# Makefile raiz - compila todas as versões

.PHONY: all clean compile-all benchmark-all analyze serial openmp cuda mpi hybrid

all: compile-all

# Compila todas as versões
compile-all: serial openmp cuda mpi hybrid

serial:
	$(MAKE) -C serial

openmp:
	$(MAKE) -C openmp

cuda:
	$(MAKE) -C cuda

mpi:
	$(MAKE) -C mpi

hybrid:
	$(MAKE) -C hybrid

# Gera datasets se não existirem
data:
	cd data && bash generate_datasets.sh

# Roda todos os benchmarks
benchmark-all: compile-all data
	bash scripts/benchmark_all.sh

# Gera análises e gráficos
analyze: benchmark-all
	python3 analysis/analyze_speedup.py
	python3 analysis/analyze_scalability.py
	python3 analysis/analyze_efficiency.py
	python3 analysis/generate_report.py

# Limpeza
clean:
	$(MAKE) -C serial clean
	$(MAKE) -C openmp clean
	$(MAKE) -C cuda clean
	$(MAKE) -C mpi clean
	$(MAKE) -C hybrid clean
	rm -rf results/*.txt report/figures/*.png

# Limpeza completa (inclui dados gerados)
distclean: clean
	rm -f data/*.csv
	rm -rf results/ report/figures/

# Help
help:
	@echo "Comandos disponíveis:"
	@echo "  make compile-all    - Compila todas as versões"
	@echo "  make data           - Gera datasets de teste"
	@echo "  make benchmark-all  - Roda todos os benchmarks"
	@echo "  make analyze        - Gera análises e relatórios"
	@echo "  make clean          - Remove executáveis"
	@echo "  make distclean      - Remove executáveis e dados gerados"
	@echo ""
	@echo "Alvos individuais:"
	@echo "  make serial         - Compila versão serial"
	@echo "  make openmp         - Compila versão OpenMP"
	@echo "  make cuda           - Compila versão CUDA"
	@echo "  make mpi            - Compila versão MPI"
	@echo "  make hybrid         - Compila versões híbridas"
