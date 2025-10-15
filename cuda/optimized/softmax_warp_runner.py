"""
Python Runner for Warp-Shuffle CUDA Softmax.
"""

from typing import Optional
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_SOFTMAX_WARP_SOURCE = """
#define NEG_FLT_MAX -1.0e38f
constexpr int WARP_SIZE = 32;

__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

extern "C" __global__ void softmax_warp_fp32_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int rows,
    int cols
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = in + warp_id * cols;
    float* row_out = out + warp_id * cols;

    float thread_max = NEG_FLT_MAX;
    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        thread_max = fmaxf(thread_max, row_in[c]);
    }
    float row_max = warp_reduce_max(thread_max);

    float thread_sum = 0.0f;
    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        thread_sum += __expf(row_in[c] - row_max);
    }
    float row_sum = warp_reduce_sum(thread_sum);
    float inv_sum = 1.0f / row_sum;

    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        row_out[c] = __expf(row_in[c] - row_max) * inv_sum;
    }
}
"""


def cuda_softmax_warp(x: torch.Tensor) -> torch.Tensor:
    """Execute Warp-Shuffle Numerically Stable Softmax along dim=-1."""
    assert x.is_cuda and x.dtype == torch.float32
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    rows, cols = x_2d.shape

    out = torch.empty_like(x_2d)

    warps_per_block = 4
    threads = warps_per_block * 32
    blocks = (rows + warps_per_block - 1) // warps_per_block

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_SOFTMAX_WARP_SOURCE, "softmax_warp_fp32_kernel")
    engine.launch(func, (blocks, 1, 1), (threads, 1, 1), [out, x_2d, rows, cols])
    return out.reshape(orig_shape)
