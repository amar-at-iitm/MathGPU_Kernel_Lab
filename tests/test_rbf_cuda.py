"""
Tests for Fused CUDA RBF Kernel.
"""

import pytest
import torch
from pytorch.baselines.rbf import rbf_kernel_pytorch
from cuda.optimized.rbf_kernel_runner import cuda_rbf_kernel_fused


@pytest.mark.parametrize("m,n,d", [
    (64, 64, 16),
    (128, 256, 32),
    (256, 256, 64),
    (512, 512, 128),
])
def test_cuda_rbf_fused_correctness(m, n, d):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(m, d, device="cuda", dtype=torch.float32)
    y = torch.randn(n, d, device="cuda", dtype=torch.float32)

    k_fused = cuda_rbf_kernel_fused(x, y, sigma=1.5)
    k_ref = rbf_kernel_pytorch(x, y, sigma=1.5)

    torch.testing.assert_close(k_fused, k_ref, rtol=1e-4, atol=1e-4)
