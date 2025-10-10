#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 2D Register-Blocked & Vectorized CUDA GEMM: C = alpha * (A @ B) + beta * C
void launch_gemm_reg_blocked_fp32(
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
