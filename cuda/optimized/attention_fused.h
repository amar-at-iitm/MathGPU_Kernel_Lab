#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Fused Tiled Attention Kernel with Online Softmax (FlashAttention-style)
void launch_attention_fused_fp32(
    float* O,
    const float* Q,
    const float* K,
    const float* V,
    int B, int N, int D,
    float scale,
    void* stream
);

#ifdef __cplusplus
}
#endif
