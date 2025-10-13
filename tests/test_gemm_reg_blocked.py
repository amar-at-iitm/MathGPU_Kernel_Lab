"""
Tests for Register-Blocked CUDA GEMM.
"""

import pytest
import torch
from pytorch.baselines.gemm import pytorch_gemm_reference, generate_gemm_inputs, verify_gemm
from cuda.optimized.gemm_reg_blocked_runner import cuda_gemm_reg_blocked


@pytest.mark.parametrize("m,n,k", [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (128, 256, 64),
])
def test_reg_blocked_correctness(m, n, k):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    a, b = generate_gemm_inputs(m, n, k, dtype=torch.float32, device="cuda")
    c = cuda_gemm_reg_blocked(a, b)
    metrics = verify_gemm(c, a, b, rtol=1e-4, atol=1e-4)
    assert metrics["relative_error"] < 1e-4
