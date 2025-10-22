"""
Python Runner for Fused Pairwise Distance & Gaussian RBF CUDA Kernel.
"""

from typing import Optional
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_RBF_FUSED_SOURCE = """
extern "C" __global__ void rbf_fused_fp32_kernel(
    float* __restrict__ K_out,
    const float* __restrict__ X,
    const float* __restrict__ Y,
    int M, int N, int D,
    float gamma
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float dist_sq = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float diff = X[row * D + d] - Y[col * D + d];
            dist_sq += diff * diff;
        }
        K_out[row * N + col] = __expf(-gamma * dist_sq);
    }
}
"""


def cuda_rbf_kernel_fused(x: torch.Tensor,
                          y: torch.Tensor,
                          sigma: float = 1.0) -> torch.Tensor:
    """Execute Fused Gaussian RBF CUDA Kernel."""
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float32 and y.dtype == torch.float32, "Inputs must be float32"
    m, d = x.shape
    n, d2 = y.shape
    assert d == d2, f"Feature dimension mismatch: {d} vs {d2}"

    gamma = 1.0 / (2.0 * sigma * sigma)
    k_out = torch.empty((m, n), dtype=torch.float32, device=x.device)

    bx, by = 32, 32
    grid = ((n + bx - 1) // bx, (m + by - 1) // by, 1)
    block = (bx, by, 1)

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_RBF_FUSED_SOURCE, "rbf_fused_fp32_kernel")
    engine.launch(func, grid, block, [k_out, x, y, m, n, d, gamma])
    return k_out
