"""
Test suite to verify basic test infrastructure, GPU availability, and accuracy metrics.
"""

import pytest
import torch
from numerical_analysis.accuracy import (
    relative_error,
    max_absolute_error,
    root_mean_square_error,
    snr_db,
    compute_error_metrics,
    assert_close_custom,
)


def test_cuda_device_available():
    """Verify that an NVIDIA CUDA device is detected by PyTorch."""
    assert torch.cuda.is_available(), "CUDA device must be available on this host."
    device_name = torch.cuda.get_device_name(0)
    assert len(device_name) > 0, "CUDA device name should not be empty."
    print(f"\n[Test Info] Detected CUDA Device: {device_name}")


def test_exact_identity_error():
    """Identical tensors should produce near-zero relative and absolute errors."""
    x = torch.randn(100, 100, dtype=torch.float32)
    metrics = compute_error_metrics(x, x)

    assert metrics["relative_error"] < 1e-7
    assert metrics["max_absolute_error"] < 1e-7
    assert metrics["rmse"] < 1e-7
    assert metrics["snr_db"] > 100.0


def test_known_perturbation_error():
    """Verify metrics against a known synthetic delta."""
    ref = torch.ones(10, 10, dtype=torch.float32)
    pert = ref + 0.01  # 1% perturbation

    rel_err = relative_error(ref, pert)
    assert 0.009 < rel_err < 0.011

    max_err = max_absolute_error(ref, pert)
    assert abs(max_err - 0.01) < 1e-6

    rmse = root_mean_square_error(ref, pert)
    assert abs(rmse - 0.01) < 1e-6


def test_assert_close_custom_pass():
    """assert_close_custom should succeed within tolerance."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.00001, 2.00001, 3.00001])
    assert_close_custom(a, b, rtol=1e-4, atol=1e-4)


def test_assert_close_custom_fail():
    """assert_close_custom should raise AssertionError when exceeding tolerance."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.1, 2.1, 3.1])
    with pytest.raises(AssertionError):
        assert_close_custom(a, b, rtol=1e-4, atol=1e-4)
