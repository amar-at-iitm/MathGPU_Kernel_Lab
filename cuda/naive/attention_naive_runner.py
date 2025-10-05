"""
Python Runner for Naive Attention Pipeline.
"""

import math
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine
from cuda.optimized.softmax_warp_runner import cuda_softmax_warp

_SCORES_SOURCE = """
extern "C" __global__ void attention_scores_kernel(
    float* __restrict__ S,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    int B, int N, int D,
    float scale
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (row < N && col < N) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[(b * N + row) * D + d] * K[(b * N + col) * D + d];
        }
        S[(b * N + row) * N + col] = dot * scale;
    }
}
"""


def cuda_attention_naive(q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor) -> torch.Tensor:
    """
    Execute Naive Attention:
    S = (Q @ K^T) / sqrt(D)
    P = softmax(S)
    O = P @ V
    q, k, v: (B, N, D)
    """
    B, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # 1. Compute scores matrix S (B, N, N)
    S = torch.empty((B, N, N), device=q.device, dtype=torch.float32)
    bx, by = 16, 16
    grid = ((N + bx - 1) // bx, (N + by - 1) // by, B)
    block = (bx, by, 1)

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_SCORES_SOURCE, "attention_scores_kernel")
    engine.launch(func, grid, block, [S, q, k, B, N, D, scale])

    # 2. Softmax along sequence dimension
    P = cuda_softmax_warp(S)

    # 3. Output projection: P @ V -> (B, N, D)
    O = torch.bmm(P, v)
    return O
