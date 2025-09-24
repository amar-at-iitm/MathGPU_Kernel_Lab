"""
Tests for PyTorch cuBLAS GEMM baseline reference.
"""

import pytest
import torch
from pytorch.baselines.gemm import pytorch_gemm_reference, verify_gemm, generate_gemm_inputs


@pytest.mark.parametrize("m,n,k", [
    (64, 64, 64),
    (128, 256, 64),
    (512, 512, 512),
])
def test_pytorch_gemm_fp32(m, n, k):
    """Verify FP32 GEMM matches FP64 reference within tight tolerance."""
    a, b = generate_gemm_inputs(m, n, k, dtype=torch.float32, device="cuda")
    c = pytorch_gemm_reference(a, b)
    metrics = verify_gemm(c, a, b, rtol=1e-4, atol=1e-4)
    assert metrics["relative_error"] < 1e-4


@pytest.mark.parametrize("m,n,k", [
    (128, 128, 128),
    (256, 256, 256),
])
def test_pytorch_gemm_fp16(m, n, k):
    """Verify FP16 GEMM execution and looser tolerance check."""
    a, b = generate_gemm_inputs(m, n, k, dtype=torch.float16, device="cuda")
    c = pytorch_gemm_reference(a, b)
    metrics = verify_gemm(c, a, b, rtol=1e-2, atol=1e-2)
    assert metrics["relative_error"] < 1e-2


def test_pytorch_gemm_alpha_beta():
    """Verify C = alpha * (A @ B) + beta * C."""
    a, b = generate_gemm_inputs(128, 128, 128, dtype=torch.float32, device="cuda")
    c_init = torch.ones(128, 128, dtype=torch.float32, device="cuda")
    alpha = 2.5
    beta = 1.5

    c = pytorch_gemm_reference(a, b, alpha=alpha, beta=beta, c=c_init)
    expected = alpha * torch.matmul(a, b) + beta * c_init
    torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)
