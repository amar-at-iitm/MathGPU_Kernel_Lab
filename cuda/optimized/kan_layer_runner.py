"""
Python Runner for Fused KAN RBF CUDA Kernel.
"""

import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_KAN_FORWARD_SOURCE = """
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
"""


def cuda_kan_rbf_forward(x: torch.Tensor,
                         weights: torch.Tensor,
                         mu: torch.Tensor,
                         sigma: torch.Tensor) -> torch.Tensor:
    """
    Execute Fused KAN RBF Forward Kernel.
    x: (B, I)
    weights: (I, J, K)
    mu: (K,)
    sigma: (K,)
    returns: (B, J)
    """
    assert x.is_cuda and weights.is_cuda and mu.is_cuda and sigma.is_cuda
    B, I = x.shape
    I_w, J, K = weights.shape
    assert I == I_w and mu.shape[0] == K and sigma.shape[0] == K

    inv_sigma_sq = 1.0 / (sigma ** 2)
    y = torch.empty((B, J), device=x.device, dtype=torch.float32)

    bx, by = 16, 16
    grid = ((J + bx - 1) // bx, (B + by - 1) // by, 1)
    block = (bx, by, 1)

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_KAN_FORWARD_SOURCE, "kan_rbf_forward_kernel")
    engine.launch(func, grid, block, [y, x, weights, mu, inv_sigma_sq, B, I, J, K])
    return y
