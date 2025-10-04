#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Naive Attention Pipeline:
// Un-fused matrix multiplication and softmax requiring O(N^2) memory footprint.
// ============================================================================
extern "C" __global__ void attention_scores_kernel(
    float* __restrict__ S,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    int B, int N, int D,
    float scale
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Key sequence index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Query sequence index
    int b = blockIdx.z;

    if (row < N && col < N) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[(b * N + row) * D + d] * K[(b * N + col) * D + d];
        }
        S[(b * N + row) * N + col] = dot * scale;
    }
}
