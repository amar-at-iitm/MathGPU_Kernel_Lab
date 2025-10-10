#include "gemm_reg_blocked.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

// ============================================================================
// Kernel: 2D Register-Blocked & Vectorized Matrix Multiplication
// Each thread computes a TM x TN (4x4) sub-tile stored in hardware registers.
// Thread blocks tile BM x BN (64x64) with step BK (8).
// Amortizes both global DRAM and shared memory access into register operations.
// Arithmetic Intensity: O(TM * TN) FLOPs per register fetch.
// ============================================================================
extern "C" __global__ void gemm_reg_blocked_fp32(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int N, int K,
    float alpha,
    float beta
) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
    const int total_threads = blockDim.x * blockDim.y;       // 256

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Thread coordinates inside the block tile
    const int thread_row = threadIdx.y * TM;
    const int thread_col = threadIdx.x * TN;

    float acc[TM][TN] = {0.0f};

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // Cooperatively load BM x BK elements into s_A (64 x 8 = 512 floats, 256 threads -> 2 floats/thread)
        for (int i = 0; i < (BM * BK) / total_threads; ++i) {
            int load_idx = tid + i * total_threads;
            int r = load_idx / BK;
            int c = load_idx % BK;
            int global_r = block_row + r;
            int global_c = t * BK + c;
            if (global_r < M && global_c < K) {
                s_A[r][c] = A[global_r * K + global_c];
            } else {
                s_A[r][c] = 0.0f;
            }
        }

        // Cooperatively load BK x BN elements into s_B (8 x 64 = 512 floats, 256 threads -> 2 floats/thread)
        for (int i = 0; i < (BK * BN) / total_threads; ++i) {
            int load_idx = tid + i * total_threads;
            int r = load_idx / BN;
            int c = load_idx % BN;
            int global_r = t * BK + r;
            int global_c = block_col + c;
            if (global_r < K && global_c < N) {
                s_B[r][c] = B[global_r * N + global_c];
            } else {
                s_B[r][c] = 0.0f;
            }
        }

        __syncthreads();

        // Accumulate outer products in registers
        #pragma unroll
        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            float reg_A[TM];
            float reg_B[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = s_A[thread_row + i][dot_k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[dot_k][thread_col + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back register accumulator tile to global memory C
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = block_row + thread_row + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_c = block_col + thread_col + j;
            if (global_r < M && global_c < N) {
                int out_idx = global_r * N + global_c;
                if (beta == 0.0f) {
                    C[out_idx] = alpha * acc[i][j];
                } else {
                    C[out_idx] = alpha * acc[i][j] + beta * C[out_idx];
                }
            }
        }
    }
}

extern "C" void launch_gemm_reg_blocked_fp32(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K,
    float alpha,
    float beta,
    void* stream
) {
    dim3 block(BN / TN, BM / TM); // 16 x 16 = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaStream_t s = (cudaStream_t)stream;
    gemm_reg_blocked_fp32<<<grid, block, 0, s>>>(C, A, B, M, N, K, alpha, beta);
}

#ifndef NO_STANDALONE_MAIN
int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    printf("[MathGPU Lab] Register-Blocked CUDA GEMM (%dx%dx%d)\n", M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    launch_gemm_reg_blocked_fp32(d_C, d_A, d_B, M, N, K, 1.0f, 0.0f, nullptr);
    cudaDeviceSynchronize();

    printf("[MathGPU Lab] Register-blocked kernel executed successfully.\n");
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
#endif
