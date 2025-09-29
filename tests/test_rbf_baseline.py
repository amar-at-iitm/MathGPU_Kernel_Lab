"""
Tests for PyTorch RBF baseline.
"""

import pytest
import torch
from pytorch.baselines.rbf import pairwise_sq_distance_pytorch, rbf_kernel_pytorch


def test_rbf_exact_zero_distance():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    k = rbf_kernel_pytorch(x, x, sigma=1.0)
    # Diagonal should be exactly 1.0 since distance to self is 0
    diag = torch.diag(k)
    torch.testing.assert_close(diag, torch.ones_like(diag))


def test_rbf_symmetry():
    x = torch.randn(50, 16, device="cuda")
    k = rbf_kernel_pytorch(x, x, sigma=2.0)
    torch.testing.assert_close(k, k.t())
