# Makefile para K-means MPI

CC = mpicc
CFLAGS = -O2 -std=c99 -Wall
LDFLAGS = -lm

TARGET = kmeans_1d_mpi
SRC = kmeans_1d_mpi.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
