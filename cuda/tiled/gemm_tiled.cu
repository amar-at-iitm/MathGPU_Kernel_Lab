#include "gemm_tiled.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define TILE_DIM 32

// ============================================================================
// Kernel: 2D Tiled Matrix Multiplication in Shared Memory
// Blocks stage TILE_DIM x TILE_DIM sub-matrices of A and B in high-speed SRAM,
// synchronizing with __syncthreads(). This amortizes global DRAM latency
// across all 1024 threads in the threadblock.
// Arithmetic Intensity: O(TILE_DIM) FLOPs per memory transaction.
// ============================================================================
extern "C" __global__ void gemm_tiled_fp32(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int N, int K,
    float alpha,
    float beta
) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A into shared memory
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory
        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot-products from shared memory tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta == 0.0f) {
            C[row * N + col] = alpha * acc;
        } else {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        }
    }
}

extern "C" void launch_gemm_tiled_fp32(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K,
    float alpha,
    float beta,
    void* stream
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    cudaStream_t s = (cudaStream_t)stream;
    gemm_tiled_fp32<<<grid, block, 0, s>>>(C, A, B, M, N, K, alpha, beta);
}

#ifndef NO_STANDALONE_MAIN
int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    printf("[MathGPU Lab] Standalone Tiled CUDA GEMM runner (%dx%dx%d)\n", M, N, K);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    launch_gemm_tiled_fp32(d_C, d_A, d_B, M, N, K, 1.0f, 0.0f, nullptr);
    cudaDeviceSynchronize();

    printf("[MathGPU Lab] Tiled Kernel execution completed successfully.\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
#endif
