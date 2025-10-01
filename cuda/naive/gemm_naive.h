#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Naive CUDA General Matrix Multiplication: C = alpha * (A @ B) + beta * C
// A: (M x K), B: (K x N), C: (M x N) in row-major layout
void launch_gemm_naive_fp32(
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
