"""
Pytest configuration and shared fixtures for MathGPU_Kernel_Lab.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return CUDA device if available, else skip CUDA-only tests or fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set determinism for consistent numerical verification."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def standard_matrix_shapes():
    """Collection of test matrix dimensions (M, N, K)."""
    return [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (128, 256, 64),   # Non-square
    ]


@pytest.fixture
def make_matrix_pair(device):
    """Factory fixture to create initialized random test matrices."""
    def _generator(m: int, n: int, k: int, dtype: torch.dtype = torch.float32):
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        return a, b
    return _generator
