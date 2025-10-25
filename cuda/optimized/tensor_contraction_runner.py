r"""
Python Runner for 4D Tensor Contraction CUDA Kernel.
Operation: C_{ijkl} = \sum_{a,b} A_{ijab} * B_{abkl}
"""

import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_TENSOR_CONTRACTION_SOURCE = """
#define TILE_DIM 16

extern "C" __global__ void tensor_contraction_4d_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int I, int J, int K_dim, int L,
    int A_dim, int B_dim
) {
    int M = I * J;
    int N = K_dim * L;
    int K = A_dim * B_dim;

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_k_A = t * TILE_DIM + threadIdx.x;
        if (row < M && tiled_k_A < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + tiled_k_A];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int tiled_k_B = t * TILE_DIM + threadIdx.y;
        if (tiled_k_B < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[tiled_k_B * N + col];
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
        C[row * N + col] = acc;
    }
}
"""


def cuda_tensor_contraction_4d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""
    Computes C_{ijkl} = \sum_{a,b} A_{ijab} * B_{abkl} using Tiled CUDA Kernel.
    A: (I, J, A_dim, B_dim)
    B: (A_dim, B_dim, K_dim, L)
    Returns C: (I, J, K_dim, L)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Inputs must be float32"
    assert A.ndim == 4 and B.ndim == 4, "Inputs must be 4D tensors"
    assert A.shape[2] == B.shape[0] and A.shape[3] == B.shape[1], (
        f"Contracted dimensions mismatch: A has {A.shape[2:]} but B has {B.shape[:2]}"
    )

    I, J, A_dim, B_dim = A.shape
    _, _, K_dim, L = B.shape

    M = I * J
    N = K_dim * L

    C = torch.empty((I, J, K_dim, L), dtype=torch.float32, device=A.device)

    tile_dim = 16
    grid = ((N + tile_dim - 1) // tile_dim, (M + tile_dim - 1) // tile_dim, 1)
    block = (tile_dim, tile_dim, 1)

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_TENSOR_CONTRACTION_SOURCE, "tensor_contraction_4d_kernel")
    engine.launch(func, grid, block, [C, A, B, I, J, K_dim, L, A_dim, B_dim])
    return C
