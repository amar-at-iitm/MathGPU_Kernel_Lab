"""
OpenAI Triton High-Performance 4D Tensor Contraction Kernel.
Operation: C_{ijkl} = \sum_{a,b} A_{ijab} * B_{abkl}
Features: Blocked SRAM accumulation, multi-dimensional stride support, and memory coalescing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tensor_contraction_4d_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_tensor_contraction_4d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""
    Execute 4D Tensor Contraction via OpenAI Triton:
    C_{ijkl} = \sum_{a,b} A_{ijab} * B_{abkl}
    A: shape (I, J, A_dim, B_dim)
    B: shape (A_dim, B_dim, K_dim, L)
    Returns: C shape (I, J, K_dim, L)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.ndim == 4 and B.ndim == 4, "Inputs must be 4D tensors"
    assert A.shape[2] == B.shape[0] and A.shape[3] == B.shape[1], (
        f"Contracted dimensions mismatch: {A.shape[2:]} vs {B.shape[:2]}"
    )

    I, J, A_dim, B_dim = A.shape
    _, _, K_dim, L = B.shape

    M = I * J
    K = A_dim * B_dim
    N = K_dim * L

    A_flat = A.reshape(M, K).contiguous()
    B_flat = B.reshape(K, N).contiguous()
    C_flat = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    _tensor_contraction_4d_kernel[grid](
        A_flat, B_flat, C_flat,
        M, N, K,
        A_flat.stride(0), A_flat.stride(1),
        B_flat.stride(0), B_flat.stride(1),
        C_flat.stride(0), C_flat.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )

    return C_flat.view(I, J, K_dim, L)
