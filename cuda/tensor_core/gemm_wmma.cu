#include "gemm_wmma.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 2; // 2 warps along M
constexpr int WARPS_N = 2; // 2 warps along N
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * WARP_SIZE; // 128 threads

// ============================================================================
// Kernel: Mixed-Precision Matrix Multiplication via Tensor Core WMMA
// Exploits hardware Tensor Cores using NVIDIA Warp Matrix Multiply Accumulate (WMMA).
// Inputs: FP16 (half), Accumulator / Output: FP32.
// Achieves peak theoretical Tensor TFLOPS on Ada Lovelace / Ampere architectures.
// ============================================================================
extern "C" __global__ void gemm_wmma_fp16_kernel(
    float* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M, int N, int K,
    float alpha,
    float beta
) {
    // Warp ID inside block
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    // Block row and col base in terms of 16x16 tiles
    int block_row = blockIdx.y * (WARPS_M * WMMA_M);
    int block_col = blockIdx.x * (WARPS_N * WMMA_N);

    int warp_row = block_row + warp_m * WMMA_M;
    int warp_col = block_col + warp_n * WMMA_N;

    // Boundaries check for full warp tile
    if (warp_row >= M || warp_col >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        if (k + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Scale by alpha
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] *= alpha;
    }

    // If beta != 0, load existing C and accumulate
    if (beta != 0.0f) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_init_frag;
        wmma::load_matrix_sync(c_init_frag, C + warp_row * N + warp_col, N, wmma::mem_row_major);
        for (int i = 0; i < c_frag.num_elements; ++i) {
            c_frag.x[i] += beta * c_init_frag.x[i];
        }
    }

    // Store result back to memory
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}

extern "C" void launch_gemm_wmma_fp16(
    float* C,
    const half* A,
    const half* B,
    int M, int N, int K,
    float alpha,
    float beta,
    void* stream
) {
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid((N + (WARPS_N * WMMA_N) - 1) / (WARPS_N * WMMA_N),
              (M + (WARPS_M * WMMA_M) - 1) / (WARPS_M * WMMA_M));
    cudaStream_t s = (cudaStream_t)stream;
    gemm_wmma_fp16_kernel<<<grid, block, 0, s>>>(C, A, B, M, N, K, alpha, beta);
}

#ifndef NO_STANDALONE_MAIN
int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    printf("[MathGPU Lab] Tensor Core WMMA Mixed-Precision GEMM (%dx%dx%d)\n", M, N, K);

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    launch_gemm_wmma_fp16(d_C, d_A, d_B, M, N, K, 1.0f, 0.0f, nullptr);
    cudaDeviceSynchronize();

    printf("[MathGPU Lab] Tensor Core kernel executed successfully.\n");
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
#endif
