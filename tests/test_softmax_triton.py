"""
Tests for OpenAI Triton Softmax.
"""

import pytest
import torch
from triton_kernels.softmax import triton_softmax


@pytest.mark.parametrize("rows,cols", [
    (64, 64),
    (128, 256),
    (512, 1024),
    (100, 150),
])
def test_triton_softmax_correctness(rows, cols):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    out_triton = triton_softmax(x)
    out_ref = torch.softmax(x, dim=-1)

    torch.testing.assert_close(out_triton, out_ref, rtol=1e-5, atol=1e-5)
