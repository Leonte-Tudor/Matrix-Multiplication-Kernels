#include <iostream>
#include <time.h>

#define BLOCK_DIM 8
const unsigned int TILE_DIM = 8;

template <typename T>
__global__ void mm_tiled_kernel(T* A_d, T* B_d, T* C_d, const unsigned int m, const unsigned int n, const unsigned int p){

    __shared__ T A_s[TILE_DIM][TILE_DIM];
    __shared__ T B_s[TILE_DIM][TILE_DIM];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for(unsigned int tile = 0; tile < m/TILE_DIM; ++tile){

        A_s[threadIdx.y][threadIdx.x] = A_d[row * p + tile * TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B_d[(tile * TILE_DIM + threadIdx.y) * n + col];
        __syncthreads();

        for (unsigned int i = 0; i < TILE_DIM; ++i)
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        __syncthreads();

    }

    C_d[row * n + col] = sum;

}

int main() {

    const unsigned int m = 2048, n = 1024, p = 512; // select matrix sizes (should be divisible by TILE_DIM)
    unsigned int i, c;
    auto* A = new float [m * p];
    auto* B = new float [p * n];
    auto* C = new float [m * n];

    for(i = 0; i < m * p; ++i)
        //if(i/p == i%p) // this condition will produce the identity matrix (multiplied by some scalar) when A is square
        A[i] = 3;

    for(i = 0; i < p * n; ++i)
        B[i] = 2;

    float *A_d, *B_d , *C_d;
    cudaMalloc(&A_d, m * p * sizeof(float));
    cudaMalloc(&B_d, p * n * sizeof(float));
    cudaMalloc(&C_d, m * n * sizeof(float));

    cudaMemcpy(A_d, A, m * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, p * n * sizeof(float), cudaMemcpyHostToDevice);

    const dim3 ThreadsPerBlock (BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 BlocksPerGrid (n/BLOCK_DIM, m/BLOCK_DIM, 1);

    clock_t start = clock();

    mm_tiled_kernel<<<BlocksPerGrid,ThreadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
    cudaDeviceSynchronize();

    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5f\n", elapsed);

    cudaMemcpy(C, C_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // printing useful for smaller matrices
    /*c = 1;
    for(i = 0 ; i < m * p; ++i) {
        printf("%f ", A[i]);
        if(c%p == 0)
            printf("\n");
        ++c;
    }
    printf("\n");

    c = 1;
    for(i = 0 ; i < p * n; ++i) {
        printf("%f ", B[i]);
        if(c%n == 0)
            printf("\n");
        ++c;
    }
    printf("\n");

    c = 1;
    for(i = 0 ; i < m * n; ++i) {
        printf("%f ", C[i]);
        if(c%n == 0)
            printf("\n");
        ++c;
    }*/

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    delete[] A;
    delete[] B;
    delete[] C;


}
