"""
OpenAI Triton Batched & Mixed-Precision RBF Kernel.
"""

from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def _rbf_batched_kernel(
    x_ptr, y_ptr, out_ptr,
    B, M, N, D,
    gamma,
    stride_xb, stride_xm, stride_xd,
    stride_yb, stride_yn, stride_yd,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulate squared distance over feature dimension D
    dist_sq = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d in range(0, D):
        x_ptrs = x_ptr + pid_b * stride_xb + offs_m[:, None] * stride_xm + d * stride_xd
        y_ptrs = y_ptr + pid_b * stride_yb + offs_n[None, :] * stride_yn + d * stride_yd

        x_val = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        y_val = tl.load(y_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)

        diff = x_val - y_val
        dist_sq += diff * diff

    # Evaluate Gaussian exponential
    rbf_val = tl.exp(-gamma * dist_sq)

    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, rbf_val, mask=mask_m[:, None] & mask_n[None, :])


def triton_rbf_kernel(x: torch.Tensor,
                      y: torch.Tensor,
                      sigma: float = 1.0) -> torch.Tensor:
    """
    Execute Batched & Mixed-Precision RBF Kernel in Triton.
    x: (B, M, D) or (M, D)
    y: (B, N, D) or (N, D)
    """
    is_batched = (x.dim() == 3)
    if not is_batched:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    B, M, D = x.shape
    _, N, D2 = y.shape
    assert D == D2

    gamma = 1.0 / (2.0 * sigma * sigma)
    out = torch.empty((B, M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), B)

    _rbf_batched_kernel[grid](
        x, y, out,
        B, M, N, D,
        gamma,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out if is_batched else out.squeeze(0)
