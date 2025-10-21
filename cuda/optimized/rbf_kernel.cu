#include "rbf_kernel.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Kernel: Fused Pairwise Distance & Gaussian Radial Basis Function (RBF)
// Computes K[i, j] = exp(-gamma * ||X[i] - Y[j]||^2) in a single fused pass.
// Completely bypasses materializing the O(M x N) intermediate distance matrix
// in DRAM, saving 50% to 75% of global memory traffic.
// ============================================================================
extern "C" __global__ void rbf_fused_fp32_kernel(
    float* __restrict__ K_out,
    const float* __restrict__ X,
    const float* __restrict__ Y,
    int M, int N, int D,
    float gamma
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float dist_sq = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float diff = X[row * D + d] - Y[col * D + d];
            dist_sq += diff * diff;
        }
        K_out[row * N + col] = __expf(-gamma * dist_sq);
    }
}

extern "C" void launch_rbf_fused_fp32(
    float* K,
    const float* X,
    const float* Y,
    int M, int N, int D,
    float gamma,
    void* stream
) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cudaStream_t s = (cudaStream_t)stream;
    rbf_fused_fp32_kernel<<<grid, block, 0, s>>>(K, X, Y, M, N, D, gamma);
}

#ifndef NO_STANDALONE_MAIN
int main(int argc, char** argv) {
    int M = 512, N = 512, D = 64;
    printf("[MathGPU Lab] Standalone Fused RBF Kernel (%dx%d, D=%d)\n", M, N, D);

    float *d_X, *d_Y, *d_K;
    cudaMalloc(&d_X, M * D * sizeof(float));
    cudaMalloc(&d_Y, N * D * sizeof(float));
    cudaMalloc(&d_K, M * N * sizeof(float));

    launch_rbf_fused_fp32(d_K, d_X, d_Y, M, N, D, 0.5f, nullptr);
    cudaDeviceSynchronize();

    printf("[MathGPU Lab] Fused RBF execution completed.\n");
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_K);
    return 0;
}
#endif
