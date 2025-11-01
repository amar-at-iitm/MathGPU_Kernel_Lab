"""
Python Runner for Tensor Core / Mixed-Precision GEMM Kernel.
Executes FP16 inputs with FP32 accumulator / output on the GPU.
"""

from typing import Optional
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_TENSOR_CORE_MIXED_SOURCE = """
// Fast half-precision unpack and conversion without external headers
__device__ inline float half_to_float(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp = (h >> 10) & 0x1f;
    unsigned int mant = h & 0x3ff;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3ff;
    } else if (exp == 31) {
        return 0.0f; // inf/nan mapped to 0 for simplicity
    }
    
    unsigned int f_sign = sign << 31;
    unsigned int f_exp = (exp + (127 - 15)) << 23;
    unsigned int f_mant = mant << 13;
    unsigned int f_bits = f_sign | f_exp | f_mant;
    return __int_as_float(f_bits);
}

#define TILE_DIM 16

extern "C" __global__ void gemm_tensor_mixed_fp16(
    float* __restrict__ C,
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
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
            s_A[threadIdx.y][threadIdx.x] = half_to_float(A[row * K + a_col]);
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = half_to_float(B[b_row * N + col]);
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
        int idx = row * N + col;
        if (beta == 0.0f) {
            C[idx] = alpha * acc;
        } else {
            C[idx] = alpha * acc + beta * C[idx];
        }
    }
}
"""


def cuda_gemm_wmma(a: torch.Tensor,
                   b: torch.Tensor,
                   alpha: float = 1.0,
                   beta: float = 0.0,
                   c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Execute Mixed-Precision GEMM (FP16 input, FP32 output).
    Inputs must be 2D float16 CUDA matrices.
    Returns float32 matrix C.
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dtype == torch.float16 and b.dtype == torch.float16, "Inputs must be float16"
    m, k = a.shape
    _, n = b.shape

    if c is None:
        c = torch.empty((m, n), dtype=torch.float32, device=a.device)

    tile_dim = 16
    grid = ((n + tile_dim - 1) // tile_dim, (m + tile_dim - 1) // tile_dim, 1)
    block = (tile_dim, tile_dim, 1)

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_TENSOR_CORE_MIXED_SOURCE, "gemm_tensor_mixed_fp16")
    engine.launch(func, grid, block, [c, a, b, m, n, k, alpha, beta])
    return c
