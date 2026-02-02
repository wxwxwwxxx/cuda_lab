//
// Created by bci on 2/2/26.
//
#include <cuda_runtime.h>
#include <iostream>

#define TILE 16

__global__ void matmul(
        const float* A,
        const float* B,
        float* C,
        int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < N / TILE; k++)
    {
        As[threadIdx.y][threadIdx.x] =
                A[row * N + k * TILE + threadIdx.x];

        Bs[threadIdx.y][threadIdx.x] =
                B[(k * TILE + threadIdx.y) * N + col];

        __syncthreads();

        for (int i = 0; i < TILE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main()
{
    const int N = 512;
    size_t size = N * N * sizeof(float);

    float *hA = new float[N * N];
    float *hB = new float[N * N];
    float *hC = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        hA[i] = 0.8f;
        hB[i] = 0.6f;
    }

    float *dA, *dB, *dC;

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    matmul<<<grid, block>>>(dA, dB, dC, N);

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    std::cout << "C[0] = " << hC[0] << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}