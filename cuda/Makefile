NVCC = nvcc
CFLAGS = -O2
TARGET = kmeans_1d_cuda
SOURCES = kmeans_1d_cuda.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: all clean
