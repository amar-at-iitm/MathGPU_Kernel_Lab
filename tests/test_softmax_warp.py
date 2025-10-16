"""
Tests for Warp-Shuffle Numerically Stable CUDA Softmax.
"""

import pytest
import torch
from cuda.optimized.softmax_warp_runner import cuda_softmax_warp


@pytest.mark.parametrize("rows,cols", [
    (64, 64),
    (128, 256),
    (512, 1024),
    (100, 150),
])
def test_softmax_warp_correctness(rows, cols):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    out_cuda = cuda_softmax_warp(x)
    out_ref = torch.softmax(x, dim=-1)

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-5, atol=1e-5)


def test_softmax_numerical_stability():
    """Verify safe softmax does not produce NaNs with large input numbers."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Inputs that would cause standard exp() to overflow float32 (exp(1000) = inf)
    x = torch.tensor([[1000.0, 1001.0, 999.0], [-500.0, -501.0, -499.0]], device="cuda")
    out_cuda = cuda_softmax_warp(x)
    out_ref = torch.softmax(x, dim=-1)

    assert not torch.isnan(out_cuda).any()
    assert not torch.isinf(out_cuda).any()
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-5, atol=1e-5)
