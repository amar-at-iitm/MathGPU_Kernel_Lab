#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 4D Tensor Contraction: C_{ijkl} = sum_{a,b} A_{ijab} * B_{abkl}
void launch_tensor_contraction_4d(
    float* C,
    const float* A,
    const float* B,
    int I, int J, int K_dim, int L, int A_dim, int B_dim,
    void* stream
);

#ifdef __cplusplus
}
#endif
