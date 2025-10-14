#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Warp-Shuffle Numerically Stable Row-wise Softmax
void launch_softmax_warp_fp32(
    float* out,
    const float* in,
    int rows,
    int cols,
    void* stream
);

#ifdef __cplusplus
}
#endif
