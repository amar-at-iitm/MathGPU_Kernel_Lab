"""
Tests for Attention Baselines (PyTorch vs. Naive CUDA).
"""

import pytest
import torch
from pytorch.baselines.attention import scaled_dot_product_attention_pytorch
from cuda.naive.attention_naive_runner import cuda_attention_naive


@pytest.mark.parametrize("B,N,D", [
    (2, 64, 32),
    (4, 128, 64),
    (1, 256, 64),
])
def test_attention_naive_vs_pytorch(B, N, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    q = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, D, device="cuda", dtype=torch.float32)

    o_ref = scaled_dot_product_attention_pytorch(q, k, v)
    o_cuda = cuda_attention_naive(q, k, v)

    torch.testing.assert_close(o_cuda, o_ref, rtol=1e-4, atol=1e-4)
