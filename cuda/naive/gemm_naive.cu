#include "gemm_naive.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Kernel: Naive General Matrix Multiplication in FP32
// Each thread computes one element C[row, col] by reading entire rows of A
// and columns of B directly from global memory without shared memory reuse.
// Arithmetic Intensity: O(1) FLOPs per memory access -> highly memory-bound.
// ============================================================================
extern "C" __global__ void gemm_naive_fp32(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int N, int K,
    float alpha,
    float beta
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }

        if (beta == 0.0f) {
            C[row * N + col] = alpha * acc;
        } else {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        }
    }
}

extern "C" void launch_gemm_naive_fp32(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K,
    float alpha,
    float beta,
    void* stream
) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cudaStream_t s = (cudaStream_t)stream;
    gemm_naive_fp32<<<grid, block, 0, s>>>(C, A, B, M, N, K, alpha, beta);
}

#ifndef NO_STANDALONE_MAIN
int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    printf("[MathGPU Lab] Standalone Naive CUDA GEMM runner (%dx%dx%d)\n", M, N, K);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    launch_gemm_naive_fp32(d_C, d_A, d_B, M, N, K, 1.0f, 0.0f, nullptr);
    cudaDeviceSynchronize();

    printf("[MathGPU Lab] Kernel execution completed successfully.\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
#endif
