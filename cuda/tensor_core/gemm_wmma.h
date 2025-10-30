#pragma once
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tensor Core GEMM via NVIDIA WMMA API (FP16 inputs, FP32 accumulator)
void launch_gemm_wmma_fp16(
    float* C,
    const half* A,
    const half* B,
    int M, int N, int K,
    float alpha,
    float beta,
    void* stream
);

#ifdef __cplusplus
}
#endif
