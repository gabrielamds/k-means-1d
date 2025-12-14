# Makefile para K-means Serial (Baseline)

CC = gcc
CFLAGS = -O2 -std=c99 -Wall
LDFLAGS = -lm

TARGET = kmeans_1d_naive
SRC = kmeans_1d_naive.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
