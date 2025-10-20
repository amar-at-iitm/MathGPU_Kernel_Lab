"""
Tests for Fused Tiled CUDA Attention (FlashAttention style).
"""

import pytest
import torch
from pytorch.baselines.attention import scaled_dot_product_attention_pytorch
from cuda.optimized.attention_fused_runner import cuda_attention_fused


@pytest.mark.parametrize("B,N,D", [
    (2, 64, 32),
    (4, 128, 64),
    (1, 256, 64),
    (2, 512, 64),
])
def test_attention_fused_vs_pytorch(B, N, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    q = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, D, device="cuda", dtype=torch.float32)

    o_ref = scaled_dot_product_attention_pytorch(q, k, v)
    o_fused = cuda_attention_fused(q, k, v)

    torch.testing.assert_close(o_fused, o_ref, rtol=1e-4, atol=1e-4)
