#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Fused Pairwise Distance and Gaussian RBF Kernel
void launch_rbf_fused_fp32(
    float* K,
    const float* X,
    const float* Y,
    int M, int N, int D,
    float gamma,
    void* stream
);

#ifdef __cplusplus
}
#endif
