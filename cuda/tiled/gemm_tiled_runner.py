"""
Python Runner for 2D Tiled Shared-Memory CUDA GEMM Kernel.
Compiles and executes cuda/tiled/gemm_tiled.cu on the GPU.
"""

import os
from typing import Optional
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_TILED_KERNEL_SOURCE = """
#define TILE_DIM 32

extern "C" __global__ void gemm_tiled_fp32(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int N, int K,
    float alpha,
    float beta
) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta == 0.0f) {
            C[row * N + col] = alpha * acc;
        } else {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        }
    }
}
"""


def cuda_gemm_tiled(a: torch.Tensor,
                    b: torch.Tensor,
                    alpha: float = 1.0,
                    beta: float = 0.0,
                    c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Execute 2D Tiled Shared-Memory CUDA GEMM: C = alpha * (A @ B) + beta * C.
    Both A and B must be 2D float32 CUDA tensors.
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Only float32 is supported"
    assert a.dim() == 2 and b.dim() == 2, "A and B must be 2D matrices"
    assert a.size(1) == b.size(0), f"Dimension mismatch: A is {a.shape}, B is {b.shape}"

    m, k = a.shape
    _, n = b.shape

    if c is None:
        c = torch.empty((m, n), dtype=torch.float32, device=a.device)
    else:
        assert c.is_cuda and c.shape == (m, n)

    tile_dim = 32
    grid_dim_x = (n + tile_dim - 1) // tile_dim
    grid_dim_y = (m + tile_dim - 1) // tile_dim

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_TILED_KERNEL_SOURCE, "gemm_tiled_fp32")

    engine.launch(
        func,
        grid=(grid_dim_x, grid_dim_y, 1),
        block=(tile_dim, tile_dim, 1),
        args=[c, a, b, m, n, k, alpha, beta]
    )
    return c
