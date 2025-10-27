#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Kernel: Fused KAN Gaussian RBF Forward Pass
// Computes y_bj = sum_{i=1}^I sum_{k=1}^K c_{ijk} * exp(-0.5 * (x_{bi} - mu_k)^2 / sigma_k^2)
// Direct kernel fusion prevents creating large intermediate (B, I, K) tensors.
// ============================================================================
extern "C" __global__ void kan_rbf_forward_kernel(
    float* __restrict__ Y,
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ Mu,
    const float* __restrict__ InvSigmaSq,
    int B, int I, int J, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && j < J) {
        float total = 0.0f;
        for (int i = 0; i < I; ++i) {
            float x_val = X[b * I + i];
            for (int k = 0; k < K; ++k) {
                float diff = x_val - Mu[k];
                float basis = __expf(-0.5f * diff * diff * InvSigmaSq[k]);
                float w_val = W[(i * J + j) * K + k];
                total += basis * w_val;
            }
        }
        Y[b * J + j] = total;
    }
}
