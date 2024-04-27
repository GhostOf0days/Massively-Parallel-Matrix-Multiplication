/*
    Parallel Programming Matrix Multiplication.
    Serial version of matrix multiplication in C.
    This will not work on a normal computer due to clockcycle.h.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "clockcycle.h"

// This function intializes a matrix of the specified size with random numbers
void matrixInit(float* matrix, int size, unsigned int seed, int offset) {
    srand(seed + offset);
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / 10000;
    }
}

// Display matrix in table with rows and columns.
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Serial implementation of matrix multiplication
void matrixMultSerial(float* matrixOne, float* matrixTwo, float* resultMatrix, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                // Add to runing sum to be set in reuslt matrix
                sum += matrixOne[i * n + l] * matrixTwo[l * k + j];
            }
            resultMatrix[i * k + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    // Check commandline arguments
    if (argc != 4) {
        printf("Incorrect number of . Usage: %s <m> <n> <k>\n", argv[0]);
        printf("m: number of rows in matrix one\n");
        printf("n: number of columns in matrix one and rows in matrix two\n");
        printf("k: number of columns in matrix two\n");
        return 1;
    }

    // Parse command line arguments.
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // Allocate memory for matrices
    float *matrixOne = (float*)malloc(m * n * sizeof(float));
    float *matrixTwo = (float*)malloc(n * k * sizeof(float));
    float *resultMatrix = (float*)malloc(m * k * sizeof(float));

    // Initialize matrices
    matrixInit(matrixOne, m * n, 0, 0);
    matrixInit(matrixTwo, n * k, 0, 1);

    // Start timer
    unsigned long long start_clock = clock_now();

    // Perform computation
    matrixMultSerial(matrixOne, matrixTwo, resultMatrix, m, n, k);

    // Stop timer
    unsigned long long end_clock = clock_now();

    double total_time = (((double) end_clock) - ((double) start_clock))/512000000;


    // Make sure the matrix prints only at reasonable sizes
    if (m * k < 256) {
        printf("Matrix one:\n");
        printMatrix(matrixOne, m, n);
        printf("Matrix two:\n");
        printMatrix(matrixTwo, n, k);
        printf("Result matrix:\n");
        printMatrix(resultMatrix, m, k);
    }

    // Print timing
    printf("Time from provided clockcycle.h file: %f seconds\n", total_time);

    // Free memory
    free(matrixOne);
    free(matrixTwo);
    free(resultMatrix);

    return 0;
}
