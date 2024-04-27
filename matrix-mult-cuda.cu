/* 
    Parallel Programming Matrix Multiplication.
    Parallelized version of matrix multiplication in CUDA and MPI.
    CUDA Code
*/

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void matrix_mult_kernel(float* matrixOne, float* matrixTwo, float* resultMatrix, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            // Calculate the running sum for the current result matrix position
            sum += matrixOne[row * n + i] * matrixTwo[i * k + col];
        }
        resultMatrix[row * k + col] = sum;
    }
}

extern "C" {
    void matrix_mult_cuda(float* matrixOne, float* matrixTwo, float* resultMatrix, int m, int n, int k) {
        // Alocate memory for 3 matrices
        float* d_matrixOne, * d_matrixTwo, * d_resultMatrix;
        cudaMallocManaged(&d_matrixOne, m * n * sizeof(float));
        cudaMallocManaged(&d_matrixTwo, n * k * sizeof(float));
        cudaMallocManaged(&d_resultMatrix, m * k * sizeof(float));

        // Copy the values of the 2 given matricies from device to host
        cudaMemcpy(d_matrixOne, matrixOne, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixTwo, matrixTwo, n * k * sizeof(float), cudaMemcpyHostToDevice);

        // Setting up the grid size and call the CUDA function to do matrix multiplications
        int blockSize = BLOCK_SIZE;
        // See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3. Integer vector type to specify dimensions. Initializes unspecified components to 1.
        dim3 gridSize((k + blockSize - 1) / blockSize, (m + blockSize - 1) / blockSize);
        matrix_mult_kernel<<<gridSize, dim3(blockSize, blockSize)>>>(d_matrixOne, d_matrixTwo, d_resultMatrix, m, n, k);

        // Wait until all GPU work is finished
        cudaDeviceSynchronize();

        // Copies results of the matrix multiplication from device (d_resultMatrix) to host (resultMatrix)
        cudaMemcpy(resultMatrix, d_resultMatrix, m * k * sizeof(float), cudaMemcpyDeviceToHost);

        // Free all 3 matrices
        cudaFree(d_matrixOne);
        cudaFree(d_matrixTwo);
        cudaFree(d_resultMatrix);
    }

    // Fill matrix with random values using cuRAND
    void matrix_init(float* matrix, int size, unsigned int seed, int offset) {
        curandGenerator_t gen;
        // Create random generator instance
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        // Set random generator set
        curandSetPseudoRandomGeneratorSeed(gen, seed + offset);
        // Make sure generated values follow uniform/normal ditribution
        curandGenerateUniform(gen, matrix, size);
        // End random generator instance
        curandDestroyGenerator(gen);
    }

    // Display matrix in table with rows and columns.
    void print_matrix(float* matrix, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%.2f ", matrix[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}