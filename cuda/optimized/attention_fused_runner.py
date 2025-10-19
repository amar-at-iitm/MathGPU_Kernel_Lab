"""
Python Runner for Fused Tiled CUDA Attention Kernel.
"""

import math
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine

_ATTENTION_FUSED_SOURCE = """
#define NEG_FLT_MAX -1.0e38f

extern "C" __global__ void attention_fused_fp32_kernel(
    float* __restrict__ O,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    int B, int N, int D,
    float scale
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (q_idx >= N) return;

    const float* q_row = Q + (b * N + q_idx) * D;
    float* o_row = O + (b * N + q_idx) * D;

    float m_prev = NEG_FLT_MAX;
    float l_prev = 0.0f;

    float acc_o[128];
    for (int d = 0; d < D; ++d) {
        acc_o[d] = 0.0f;
    }

    for (int k_idx = 0; k_idx < N; ++k_idx) {
        const float* k_row = K + (b * N + k_idx) * D;
        const float* v_row = V + (b * N + k_idx) * D;

        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;

        float m_new = fmaxf(m_prev, score);
        float alpha = __expf(m_prev - m_new);
        float p = __expf(score - m_new);

        float l_new = alpha * l_prev + p;

        for (int d = 0; d < D; ++d) {
            acc_o[d] = alpha * acc_o[d] + p * v_row[d];
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    float inv_l = 1.0f / l_prev;
    for (int d = 0; d < D; ++d) {
        o_row[d] = acc_o[d] * inv_l;
    }
}
"""


def cuda_attention_fused(q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor) -> torch.Tensor:
    """Execute Fused Tiled FlashAttention Kernel."""
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32
    B, N, D = q.shape
    assert D <= 128, "Fused kernel registers configured for head dimension D <= 128"

    scale = 1.0 / math.sqrt(D)
    O = torch.empty_like(q)

    threads = 128
    blocks_x = (N + threads - 1) // threads
    grid = (blocks_x, B, 1)
    block = (threads, 1, 1)

    engine = get_cuda_jit_engine()
    func = engine.compile_and_load(_ATTENTION_FUSED_SOURCE, "attention_fused_fp32_kernel")
    engine.launch(func, grid, block, [O, q, k, v, B, N, D, scale])
    return O
