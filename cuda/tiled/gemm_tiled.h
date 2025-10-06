#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 2D Tiled CUDA GEMM with Shared Memory: C = alpha * (A @ B) + beta * C
void launch_gemm_tiled_fp32(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K,
    float alpha,
    float beta,
    void* stream
);

#ifdef __cplusplus
}
#endif
