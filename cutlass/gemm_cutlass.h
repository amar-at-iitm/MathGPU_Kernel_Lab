#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// CUTLASS-style structured Device GEMM interface
// Configured with threadblock tile, warp tile, and epilogue
void launch_cutlass_gemm_fp32(
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
