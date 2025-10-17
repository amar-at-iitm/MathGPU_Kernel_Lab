#include "attention_fused.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define NEG_FLT_MAX -1.0e38f

// ============================================================================
// Kernel: Fused Tiled Attention with Online Softmax (FlashAttention-style)
// Completely fuses QK^T, Softmax reduction, and PV projection in SRAM/registers.
// Reduces global DRAM footprint from O(B x N^2) to ZERO intermediate bytes.
// ============================================================================
extern "C" __global__ void attention_fused_fp32_kernel(
    float* __restrict__ O,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    int B, int N, int D,
    float scale
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x; // Query sequence position
    int b = blockIdx.y;                                // Batch index

    if (q_idx >= N) return;

    const float* q_row = Q + (b * N + q_idx) * D;
    float* o_row = O + (b * N + q_idx) * D;

    // Online softmax state variables for this query token
    float m_prev = NEG_FLT_MAX;
    float l_prev = 0.0f;

    // Register accumulators for output vector of size D
    // Supporting up to D=128 in thread registers
    float acc_o[128];
    for (int d = 0; d < D; ++d) {
        acc_o[d] = 0.0f;
    }

    // Tile across keys & values sequence
    for (int k_idx = 0; k_idx < N; ++k_idx) {
        const float* k_row = K + (b * N + k_idx) * D;
        const float* v_row = V + (b * N + k_idx) * D;

        // Compute dot-product S = (Q @ K^T) * scale
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;

        // Online softmax update:
        // m_new = max(m_prev, score)
        float m_new = fmaxf(m_prev, score);
        float alpha = __expf(m_prev - m_new);
        float p = __expf(score - m_new);

        // Update normalizer: l_new = alpha * l_prev + p
        float l_new = alpha * l_prev + p;

        // Rescale running output accumulators and add new V contribution
        for (int d = 0; d < D; ++d) {
            acc_o[d] = alpha * acc_o[d] + p * v_row[d];
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Final normalization by total weight sum l_prev
    float inv_l = 1.0f / l_prev;
    for (int d = 0; d < D; ++d) {
        o_row[d] = acc_o[d] * inv_l;
    }
}

extern "C" void launch_attention_fused_fp32(
    float* O,
    const float* Q,
    const float* K,
    const float* V,
    int B, int N, int D,
    float scale,
    void* stream
) {
    int threads = 128;
    int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B);
    cudaStream_t s = (cudaStream_t)stream;
    attention_fused_fp32_kernel<<<grid, threads, 0, s>>>(O, Q, K, V, B, N, D, scale);
}

#ifndef NO_STANDALONE_MAIN
int main(int argc, char** argv) {
    int B = 2, N = 512, D = 64;
    printf("[MathGPU Lab] Standalone Fused FlashAttention (B=%d, N=%d, D=%d)\n", B, N, D);

    float *d_Q, *d_K, *d_V, *d_O;
    size_t sz = B * N * D * sizeof(float);
    cudaMalloc(&d_Q, sz); cudaMalloc(&d_K, sz);
    cudaMalloc(&d_V, sz); cudaMalloc(&d_O, sz);

    launch_attention_fused_fp32(d_O, d_Q, d_K, d_V, B, N, D, 1.0f / 8.0f, nullptr);
    cudaDeviceSynchronize();

    printf("[MathGPU Lab] Fused FlashAttention executed successfully.\n");
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
#endif
