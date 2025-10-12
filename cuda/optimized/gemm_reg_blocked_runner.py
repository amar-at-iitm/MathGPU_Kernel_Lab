"""
Python Runner for 2D Register-Blocked CUDA GEMM Kernel.
Compiles and executes cuda/optimized/gemm_reg_blocked.cu.
"""

from typing import Optional
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_REG_BLOCKED_SOURCE = """
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

extern "C" __global__ void gemm_reg_blocked_fp32(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int N, int K,
    float alpha,
    float beta
) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int thread_row = threadIdx.y * TM;
    const int thread_col = threadIdx.x * TN;

    float acc[TM][TN] = {0.0f};
    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        for (int i = 0; i < (BM * BK) / total_threads; ++i) {
            int load_idx = tid + i * total_threads;
            int r = load_idx / BK;
            int c = load_idx % BK;
            int global_r = block_row + r;
            int global_c = t * BK + c;
            s_A[r][c] = (global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f;
        }

        for (int i = 0; i < (BK * BN) / total_threads; ++i) {
            int load_idx = tid + i * total_threads;
            int r = load_idx / BN;
            int c = load_idx % BN;
            int global_r = t * BK + r;
            int global_c = block_col + c;
            s_B[r][c] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            float reg_A[TM];
            float reg_B[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = s_A[thread_row + i][dot_k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[dot_k][thread_col + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = block_row + thread_row + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_c = block_col + thread_col + j;
            if (global_r < M && global_c < N) {
                int out_idx = global_r * N + global_c;
                C[out_idx] = (beta == 0.0f) ? (alpha * acc[i][j]) : (alpha * acc[i][j] + beta * C[out_idx]);
            }
        }
    }
}
"""


def cuda_gemm_reg_blocked(a: torch.Tensor,
                          b: torch.Tensor,
                          alpha: float = 1.0,
                          beta: float = 0.0,
                          c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Execute 2D Register-Blocked CUDA GEMM."""
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Inputs must be float32"
    m, k = a.shape
    _, n = b.shape

    if c is None:
        c = torch.empty((m, n), dtype=torch.float32, device=a.device)

    bm, bn = 64, 64
    tm, tn = 4, 4
    block = (bn // tn, bm // tm, 1) # 16 x 16 = 256 threads
    grid = ((n + bn - 1) // bn, (m + bm - 1) // bm, 1)

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_REG_BLOCKED_SOURCE, "gemm_reg_blocked_fp32")
    engine.launch(func, grid, block, [c, a, b, m, n, k, alpha, beta])
    return c
