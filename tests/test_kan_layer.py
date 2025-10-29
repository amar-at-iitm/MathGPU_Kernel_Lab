"""
Tests for KAN RBF Layer (PyTorch Reference vs. CUDA Fused vs. Triton).
"""

import pytest
import torch
from pytorch.baselines.kan_layer import PyTorchKANRBFLayer
from cuda.optimized.kan_layer_runner import cuda_kan_rbf_forward
from triton_kernels.kan import triton_kan_rbf_forward


def test_kan_layer_cuda_vs_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, I, J, K = 32, 16, 8, 5
    layer = PyTorchKANRBFLayer(in_features=I, out_features=J, num_bases=K).cuda()

    x = torch.randn(B, I, device="cuda", dtype=torch.float32)

    # PyTorch reference forward
    y_ref = layer(x)

    # CUDA kernel forward
    y_cuda = cuda_kan_rbf_forward(x, layer.weights, layer.grid_mu, layer.grid_sigma)

    torch.testing.assert_close(y_cuda, y_ref, rtol=1e-4, atol=1e-4)


def test_kan_layer_triton_vs_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, I, J, K = 32, 16, 8, 5
    layer = PyTorchKANRBFLayer(in_features=I, out_features=J, num_bases=K).cuda()

    x = torch.randn(B, I, device="cuda", dtype=torch.float32)

    y_ref = layer(x)
    y_triton = triton_kan_rbf_forward(x, layer.weights, layer.grid_mu, layer.grid_sigma)

    torch.testing.assert_close(y_triton, y_ref, rtol=1e-4, atol=1e-4)
