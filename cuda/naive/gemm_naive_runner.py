"""
Python Runner for Naive CUDA GEMM Kernel.
Compiles and executes cuda/naive/gemm_naive.cu on the GPU.
"""

import os
from typing import Optional
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_NAIVE_KERNEL_SOURCE = """
extern "C" __global__ void gemm_naive_fp32(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int N, int K,
    float alpha,
    float beta
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }

        if (beta == 0.0f) {
            C[row * N + col] = alpha * acc;
        } else {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        }
    }
}
"""


def cuda_gemm_naive(a: torch.Tensor,
                    b: torch.Tensor,
                    alpha: float = 1.0,
                    beta: float = 0.0,
                    c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Execute Naive CUDA GEMM: C = alpha * (A @ B) + beta * C.
    Both A and B must be 2D float32 CUDA tensors.
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Only float32 is supported for naive GEMM"
    assert a.dim() == 2 and b.dim() == 2, "A and B must be 2D matrices"
    assert a.size(1) == b.size(0), f"Dimension mismatch: A is {a.shape}, B is {b.shape}"

    m, k = a.shape
    _, n = b.shape

    if c is None:
        c = torch.empty((m, n), dtype=torch.float32, device=a.device)
    else:
        assert c.is_cuda and c.shape == (m, n)

    # 32x32 thread block = 1024 threads
    block_dim_x = 32
    block_dim_y = 32
    grid_dim_x = (n + block_dim_x - 1) // block_dim_x
    grid_dim_y = (m + block_dim_y - 1) // block_dim_y

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_NAIVE_KERNEL_SOURCE, "gemm_naive_fp32")

    engine.launch(
        func,
        grid=(grid_dim_x, grid_dim_y, 1),
        block=(block_dim_x, block_dim_y, 1),
        args=[c, a, b, m, n, k, alpha, beta]
    )
    return c
