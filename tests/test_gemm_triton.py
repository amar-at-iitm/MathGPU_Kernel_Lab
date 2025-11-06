import pytest
import torch
from triton_kernels.gemm import triton_gemm


@pytest.mark.parametrize("m,n,k", [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (128, 256, 64),
])
def test_triton_gemm_correctness(m, n, k):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Scaled inputs to prevent large accumulation drift
    scale = 1.0 / (k ** 0.5)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32) * scale
    b = torch.randn(k, n, device="cuda", dtype=torch.float32) * scale

    c_triton = triton_gemm(a, b)
    c_ref = torch.matmul(a, b)

    # Allow TF32 tolerance on Tensor Core architectures
    torch.testing.assert_close(c_triton, c_ref, rtol=1e-2, atol=1e-2)
