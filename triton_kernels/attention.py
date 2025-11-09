"""
OpenAI Triton FlashAttention Forward Kernel with Block Tiling & Online Softmax.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attention_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Online softmax accumulators
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over key/value blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        curr_offs_n = start_n + offs_n
        k = tl.load(k_ptrs + start_n * stride_kn, mask=curr_offs_n[:, None] < N_CTX, other=0.0)

        # qk: (BLOCK_M, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Mask padding
        qk = tl.where(curr_offs_n[None, :] < N_CTX, qk, -float("inf"))

        # Online softmax update
        m_curr = tl.maximum(m_prev, tl.max(qk, 1))
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.exp(m_prev - m_curr) * l_prev + tl.sum(p, 1)

        alpha = tl.exp(m_prev - m_curr)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs + start_n * stride_vn, mask=curr_offs_n[:, None] < N_CTX, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        m_prev = m_curr
        l_prev = l_curr

    # Epilogue: final normalization
    acc = acc / l_prev[:, None]
    tl.store(o_ptrs, acc.to(tl.float32), mask=offs_m[:, None] < N_CTX)


def triton_flash_attention(q: torch.Tensor,
                           k: torch.Tensor,
                           v: torch.Tensor) -> torch.Tensor:
    """
    Execute FlashAttention in Triton.
    q, k, v: (B, H, N, D) or (B, N, D)
    """
    is_3d = (q.dim() == 3)
    if is_3d:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    Z, H, N_CTX, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = triton.next_power_of_2(D)

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _fused_attention_kernel[grid](
        q, k, v, sm_scale,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
    )

    return out.squeeze(1) if is_3d else out
