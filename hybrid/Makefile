# Makefile para implementações híbridas

# OpenMP + CUDA
kmeans_1d_omp_cuda: kmeans_1d_omp_cuda.cu
	nvcc -O2 -Xcompiler -fopenmp kmeans_1d_omp_cuda.cu -o kmeans_1d_omp_cuda -lm

# OpenMP + MPI
kmeans_1d_omp_mpi: kmeans_1d_omp_mpi.c
	mpicc -O2 -std=c99 -fopenmp kmeans_1d_omp_mpi.c -o kmeans_1d_omp_mpi -lm

# MPI + CUDA
kmeans_1d_mpi_cuda: kmeans_1d_mpi_cuda.cu
	nvcc -O2 -I/usr/include/openmpi-x86_64 -L/usr/lib64/openmpi/lib kmeans_1d_mpi_cuda.cu -o kmeans_1d_mpi_cuda -lmpi

all: kmeans_1d_omp_cuda kmeans_1d_omp_mpi kmeans_1d_mpi_cuda

clean:
	rm -f kmeans_1d_omp_cuda kmeans_1d_omp_mpi kmeans_1d_mpi_cuda *.o

.PHONY: all clean
