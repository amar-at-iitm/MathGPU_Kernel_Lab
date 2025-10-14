#include "softmax_warp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

constexpr int WARP_SIZE = 32;

__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    // Broadcast maximum to all 32 threads in the warp
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    // Broadcast sum to all 32 threads in the warp
    return __shfl_sync(0xffffffff, val, 0);
}

// ============================================================================
// Kernel: Warp-Shuffle Row-Wise Safe Softmax
// Each warp of 32 threads computes softmax for one complete row.
// Eliminates shared-memory bank conflicts by communicating registers across warps.
// ============================================================================
extern "C" __global__ void softmax_warp_fp32_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int rows,
    int cols
) {
    // Warp index across grid
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = in + warp_id * cols;
    float* row_out = out + warp_id * cols;

    // Pass 1: Find maximum element across row
    float thread_max = -FLT_MAX;
    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        thread_max = fmaxf(thread_max, row_in[c]);
    }
    float row_max = warp_reduce_max(thread_max);

    // Pass 2: Compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        thread_sum += __expf(row_in[c] - row_max);
    }
    float row_sum = warp_reduce_sum(thread_sum);
    float inv_sum = 1.0f / row_sum;

    // Pass 3: Normalize and write output
    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        row_out[c] = __expf(row_in[c] - row_max) * inv_sum;
    }
}

extern "C" void launch_softmax_warp_fp32(
    float* out,
    const float* in,
    int rows,
    int cols,
    void* stream
) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * WARP_SIZE; // 128
    int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    cudaStream_t s = (cudaStream_t)stream;
    softmax_warp_fp32_kernel<<<blocks, THREADS, 0, s>>>(out, in, rows, cols);
}

#ifndef NO_STANDALONE_MAIN
int main(int argc, char** argv) {
    int rows = 1024, cols = 512;
    printf("[MathGPU Lab] Standalone Warp-Shuffle Softmax (%dx%d)\n", rows, cols);

    float *d_in, *d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));

    launch_softmax_warp_fp32(d_out, d_in, rows, cols, nullptr);
    cudaDeviceSynchronize();

    printf("[MathGPU Lab] Warp Softmax executed successfully.\n");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
#endif
