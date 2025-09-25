"""
PyTorch Baseline for Numerically Stable Softmax.
"""

import torch


def pytorch_softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standard numerically safe Softmax reference using PyTorch."""
    return torch.softmax(x, dim=dim)
