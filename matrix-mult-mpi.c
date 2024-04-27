/* 
    Parallel Programming Matrix Multiplication.
    Parallelized version of matrix multiplication in CUDA and MPI.     
    MPI Code
    This will not work on a normal computer due to clockcycle.h.
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "clockcycle.h"

extern void matrix_mult_cuda(float* matrixOne, float* matrixTwo, float* resultMatrix, int m, int n, int k);
extern void matrix_init(float* matrix, int size, unsigned int seed, int offset);
extern void print_matrix(float* matrix, int rows, int cols);

/* This function is for performing read and write operations for parallel I/O
   result will be written into a file as the code is executing. */
void matrix_io(float* matrix, int size, int rank, char* filename, bool read) {
    MPI_File file;
    MPI_Status status;
    MPI_Offset offset = rank * size * sizeof(float);

    /* Open file to do parallel I/O operations
       If 'read' is true, the file is opened in read-only mode (MPI_MODE_RDONLY).
       If 'read' is false, the file is opened in write-only mode with the create flag 
       (MPI_MODE_CREATE | MPI_MODE_WRONLY), meaning it will be created if it doesn't exist. */
    MPI_File_open(MPI_COMM_WORLD, filename, read ? MPI_MODE_RDONLY : MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    
    // Read and write for parallel I/O operations
    if (read) {
        MPI_File_read_at(file, offset, matrix, size, MPI_FLOAT, &status);
    } else {
        MPI_File_write_at(file, offset, matrix, size, MPI_FLOAT, &status);
    }

    //Close file when done
    MPI_File_close(&file);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int m, n, k;
    float *matrixOne, *matrixTwo, *resultMatrix;
    double start_time, end_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check command line arguments
    if (argc != 4) {
        if (rank == 0) {
            printf("Incorrect number of . Usage: %s <m> <n> <k>\n", argv[0]);
            printf("m: number of rows in matrix one\n");
            printf("n: number of columns in matrix one and rows in matrix two\n");
            printf("k: number of columns in matrix two\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Parse command line arguments
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

    // Allocate memory for matrices
    matrixOne = (float*)malloc(m * n * sizeof(float));
    matrixTwo = (float*)malloc(n * k * sizeof(float));
    resultMatrix = (float*)malloc(m * k * sizeof(float));

    // Initialize matrices using cuRAND
    matrix_init(matrixOne, m * n, rank, 0);
    matrix_init(matrixTwo, n * k, rank, 1);

    // Write matrices to files using parallel I/O. Get total time spent reading and writing input matrices.
    unsigned long long io_start_clock = clock_now();
    matrix_io(matrixOne, m * n, rank, "matrix_One.bin", false);
    matrix_io(matrixTwo, n * k, rank, "matrix_Two.bin", false);
    unsigned long long io_end_clock = clock_now();
    double io_total_time = (((double) io_end_clock) - ((double) io_start_clock))/512000000;
    printf("Time spent reading and writing input matrices: %f seconds\n", io_total_time);

    // Start timer
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    unsigned long long start_clock = clock_now();

    // Perform matrix multiplication using CUDA
    matrix_mult_cuda(matrixOne, matrixTwo, resultMatrix, m, n, k);

    // End timer
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    unsigned long long end_clock = clock_now();

    double total_time = (((double) end_clock) - ((double) start_clock))/512000000;

    // Write the result matrix to a file using parallel I/O
    matrix_io(resultMatrix, m * k, rank, "matrix_Result.bin", false);

    // Print matrix one, matrix two, and the result matrix
    if (rank == 0) {
        // Make sure the matrix prints only at reasonable sizes
        if (m * k < 256){
            printf("Matrix one:\n");
            print_matrix(matrixOne, m, n);
            printf("Matrix two:\n");
            print_matrix(matrixTwo, n, k);
            printf("Result matrix:\n");
            print_matrix(resultMatrix, m, k);
        }
        
        printf("Execution time: %f seconds\n", end_time - start_time);
        printf("Time from provided clockcycle.h file: %f seconds\n", total_time);
    }

    // Free memory
    free(matrixOne);
    free(matrixTwo);
    free(resultMatrix);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}