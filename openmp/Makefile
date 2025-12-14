CC = gcc
CFLAGS = -O2 -std=c99 -Wall -fopenmp
LDFLAGS = -lm
TARGET = kmeans_1d_parallel
SOURCES = kmeans_1d_parallel.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
