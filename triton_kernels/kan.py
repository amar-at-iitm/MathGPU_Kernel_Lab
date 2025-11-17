"""
OpenAI Triton KAN Gaussian RBF Layer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _kan_forward_kernel(
    x_ptr, w_ptr, mu_ptr, inv_sig_ptr, y_ptr,
    B, I, J, K,
    stride_xb, stride_xi,
    stride_wi, stride_wj, stride_wk,
    stride_yb, stride_yj,
    BLOCK_B: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_j = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

    mask_b = offs_b < B
    mask_j = offs_j < J

    acc = tl.zeros((BLOCK_B, BLOCK_J), dtype=tl.float32)

    for i in range(0, I):
        x_ptrs = x_ptr + offs_b * stride_xb + i * stride_xi
        x_val = tl.load(x_ptrs, mask=mask_b, other=0.0)

        for k in range(0, K):
            mu_k = tl.load(mu_ptr + k)
            inv_sig_k = tl.load(inv_sig_ptr + k)

            diff = x_val - mu_k
            basis_k = tl.exp(-0.5 * diff * diff * inv_sig_k) # (BLOCK_B,)

            w_ptrs = w_ptr + i * stride_wi + offs_j * stride_wj + k * stride_wk
            w_val = tl.load(w_ptrs, mask=mask_j, other=0.0) # (BLOCK_J,)

            acc += basis_k[:, None] * w_val[None, :]

    y_ptrs = y_ptr + offs_b[:, None] * stride_yb + offs_j[None, :] * stride_yj
    tl.store(y_ptrs, acc, mask=mask_b[:, None] & mask_j[None, :])


def triton_kan_rbf_forward(x: torch.Tensor,
                           weights: torch.Tensor,
                           mu: torch.Tensor,
                           sigma: torch.Tensor) -> torch.Tensor:
    """Execute KAN RBF Forward pass in Triton."""
    B, I = x.shape
    I_w, J, K = weights.shape
    assert I == I_w

    inv_sig = 1.0 / (sigma ** 2)
    y = torch.empty((B, J), device=x.device, dtype=torch.float32)

    BLOCK_B = 16
    BLOCK_J = 16
    grid = (triton.cdiv(J, BLOCK_J), triton.cdiv(B, BLOCK_B))

    _kan_forward_kernel[grid](
        x, weights, mu, inv_sig, y,
        B, I, J, K,
        x.stride(0), x.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        y.stride(0), y.stride(1),
        BLOCK_B=BLOCK_B,
        BLOCK_J=BLOCK_J,
    )
    return y
