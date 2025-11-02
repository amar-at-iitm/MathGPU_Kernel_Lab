"""
Tests for Tensor Core WMMA mixed-precision GEMM.
"""

import pytest
import torch
from cuda.tensor_core.gemm_wmma_runner import cuda_gemm_wmma


@pytest.mark.parametrize("m,n,k", [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (64, 128, 64),
])
def test_tensor_core_wmma_correctness(m, n, k):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Inputs in float16
    a = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.1
    b = torch.randn(k, n, dtype=torch.float16, device="cuda") * 0.1

    c_wmma = cuda_gemm_wmma(a, b)
    c_ref = torch.matmul(a.float(), b.float())

    torch.testing.assert_close(c_wmma, c_ref, rtol=1e-2, atol=1e-2)
