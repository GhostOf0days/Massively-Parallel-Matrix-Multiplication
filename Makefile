CC = mpixlc
NVCC = nvcc
CFLAGS = -O3
NVCCFLAGS = -O3 -arch=sm_70
LDFLAGS = -L/usr/local/cuda-11.2/lib64/
LDLIBS = -lcurand -lcudadevrt -lcudart -lstdc++

matrix-mult: matrix-mult-mpi.o matrix-mult-cuda.o
	$(CC) $(CFLAGS) matrix-mult-mpi.o matrix-mult-cuda.o -o matrix-mult $(LDFLAGS) $(LDLIBS)

matrix-mult-mpi.o: matrix-mult-mpi.c
	$(CC) $(CFLAGS) matrix-mult-mpi.c -c -o matrix-mult-mpi.o

matrix-mult-cuda.o: matrix-mult-cuda.cu
	$(NVCC) $(NVCCFLAGS) matrix-mult-cuda.cu -c -o matrix-mult-cuda.o

clean:
	rm -f matrix-mult matrix-mult-mpi.o matrix-mult-cuda.o

.PHONY: clean