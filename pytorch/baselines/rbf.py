"""
PyTorch Baseline for Radial Basis Function (RBF) & Pairwise Distance Kernels.
Formulation:
    D_{ij} = ||x_i - y_j||^2
    K_{ij} = exp(- D_{ij} / (2 * sigma^2))
"""

from typing import Tuple, Optional
import torch


def pairwise_sq_distance_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Compute pairwise squared Euclidean distance:
    D_{ij} = ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i^T * y_j
    x: (M, D), y: (N, D) -> returns (M, N)
    """
    assert x.dim() == 2 and y.dim() == 2
    assert x.size(1) == y.size(1), "Feature dimension D must match"

    # Expanded formulation: ||x||^2 + ||y||^2 - 2 x y^T
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)       # (M, 1)
    y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True).t()   # (1, N)
    dist_sq = x_norm_sq + y_norm_sq - 2.0 * torch.matmul(x, y.t())
    return torch.clamp(dist_sq, min=0.0)


def rbf_kernel_pytorch(x: torch.Tensor,
                       y: torch.Tensor,
                       sigma: float = 1.0) -> torch.Tensor:
    r"""
    Compute Gaussian RBF Kernel:
    K_{ij} = exp(- ||x_i - y_j||^2 / (2 * sigma^2))
    """
    dist_sq = pairwise_sq_distance_pytorch(x, y)
    gamma = 1.0 / (2.0 * sigma * sigma)
    return torch.exp(-gamma * dist_sq)
