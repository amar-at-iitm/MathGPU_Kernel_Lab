"""
Tests for OpenAI Triton Batched & Mixed-Precision RBF Kernel.
"""

import pytest
import torch
from pytorch.baselines.rbf import rbf_kernel_pytorch
from triton_kernels.rbf import triton_rbf_kernel


def test_triton_rbf_2d():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(128, 32, device="cuda", dtype=torch.float32)
    y = torch.randn(128, 32, device="cuda", dtype=torch.float32)

    k_triton = triton_rbf_kernel(x, y, sigma=1.2)
    k_ref = rbf_kernel_pytorch(x, y, sigma=1.2)

    torch.testing.assert_close(k_triton, k_ref, rtol=1e-4, atol=1e-4)


def test_triton_rbf_batched():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, M, N, D = 4, 64, 64, 16
    x = torch.randn(B, M, D, device="cuda", dtype=torch.float32)
    y = torch.randn(B, N, D, device="cuda", dtype=torch.float32)

    k_triton = triton_rbf_kernel(x, y, sigma=1.0)
    for b in range(B):
        ref_b = rbf_kernel_pytorch(x[b], y[b], sigma=1.0)
        torch.testing.assert_close(k_triton[b], ref_b, rtol=1e-4, atol=1e-4)
