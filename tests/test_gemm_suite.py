"""
Comprehensive Test Suite for GEMM Kernels:
PyTorch Reference vs. Naive CUDA GEMM vs. Tiled Shared-Memory CUDA GEMM.
"""

import pytest
import torch
from pytorch.baselines.gemm import pytorch_gemm_reference, generate_gemm_inputs, verify_gemm
from cuda.naive.gemm_naive_runner import cuda_gemm_naive
from cuda.tiled.gemm_tiled_runner import cuda_gemm_tiled


@pytest.mark.parametrize("m,n,k", [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (128, 256, 64),    # Non-square
    (100, 200, 150),   # Non-multiple of 32
])
def test_naive_cuda_gemm_correctness(m, n, k):
    """Verify Naive CUDA GEMM against PyTorch cuBLAS reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    a, b = generate_gemm_inputs(m, n, k, dtype=torch.float32, device="cuda")
    c_naive = cuda_gemm_naive(a, b)

    metrics = verify_gemm(c_naive, a, b, rtol=1e-4, atol=1e-4)
    assert metrics["relative_error"] < 1e-4, f"Relative error {metrics['relative_error']} too high"


@pytest.mark.parametrize("m,n,k", [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (128, 256, 64),    # Non-square
    (100, 200, 150),   # Non-multiple of 32 tile size
])
def test_tiled_cuda_gemm_correctness(m, n, k):
    """Verify Tiled Shared-Memory CUDA GEMM against PyTorch cuBLAS reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    a, b = generate_gemm_inputs(m, n, k, dtype=torch.float32, device="cuda")
    c_tiled = cuda_gemm_tiled(a, b)

    metrics = verify_gemm(c_tiled, a, b, rtol=1e-4, atol=1e-4)
    assert metrics["relative_error"] < 1e-4, f"Relative error {metrics['relative_error']} too high"


def test_tiled_gemm_alpha_beta():
    """Verify Tiled GEMM with alpha and beta scaling."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m, n, k = 128, 128, 128
    a, b = generate_gemm_inputs(m, n, k, dtype=torch.float32, device="cuda")
    c_init = torch.ones(m, n, dtype=torch.float32, device="cuda")
    alpha = 1.8
    beta = 0.5

    c_tiled = cuda_gemm_tiled(a, b, alpha=alpha, beta=beta, c=c_init.clone())
    c_ref = pytorch_gemm_reference(a, b, alpha=alpha, beta=beta, c=c_init.clone())

    torch.testing.assert_close(c_tiled, c_ref, rtol=1e-4, atol=1e-4)
