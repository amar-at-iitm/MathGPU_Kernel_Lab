"""
Tests for OpenAI Triton FlashAttention.
"""

import pytest
import torch
from pytorch.baselines.attention import scaled_dot_product_attention_pytorch
from triton_kernels.attention import triton_flash_attention


@pytest.mark.parametrize("B,N,D", [
    (2, 64, 32),
    (4, 128, 64),
    (1, 256, 64),
])
def test_triton_flash_attention_correctness(B, N, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    q = torch.randn(B, N, D, device="cuda", dtype=torch.float32) * 0.1
    k = torch.randn(B, N, D, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, N, D, device="cuda", dtype=torch.float32) * 0.1

    o_ref = scaled_dot_product_attention_pytorch(q, k, v)
    o_triton = triton_flash_attention(q, k, v)

    torch.testing.assert_close(o_triton, o_ref, rtol=1e-3, atol=1e-3)
