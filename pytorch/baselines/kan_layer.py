"""
GPU-Accelerated Kolmogorov-Arnold Network (KAN) Gaussian RBF Layer.
Mathematical formulation:
    phi_{ij}(x) = sum_{k=1}^K c_{ijk} * exp(-(x - mu_k)^2 / (2 * sigma_k^2))
    y_j = sum_{i=1}^I phi_{ij}(x_i)
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class PyTorchKANRBFLayer(nn.Module):
    """Reference PyTorch Module for KAN Layer with Gaussian RBF bases."""

    def __init__(self, in_features: int, out_features: int, num_bases: int = 8, grid_min: float = -2.0, grid_max: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bases = num_bases

        # Initialize uniform grid centers
        grid = torch.linspace(grid_min, grid_max, num_bases)
        self.register_buffer("grid_mu", grid)  # (K,)

        # Bandwidth parameter (uniform spacing)
        h = (grid_max - grid_min) / max(num_bases - 1, 1)
        sigma = torch.full((num_bases,), h)
        self.register_buffer("grid_sigma", sigma)  # (K,)

        # Trainable coefficient weights: (I, J, K)
        self.weights = nn.Parameter(torch.randn(in_features, out_features, num_bases) / math.sqrt(in_features * num_bases))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, I)
        returns y: (B, J)
        """
        B, I = x.shape
        K = self.num_bases
        # (B, I, 1) - (1, 1, K) -> (B, I, K)
        diff = x.unsqueeze(-1) - self.grid_mu.view(1, 1, K)
        basis = torch.exp(-0.5 * (diff / self.grid_sigma.view(1, 1, K)) ** 2)  # (B, I, K)

        # Tensor contraction: y_bj = sum_{i, k} basis_{bik} * weights_{ijk}
        # (B, I, K) x (I, J, K) -> (B, J)
        out = torch.einsum("bik,ijk->bj", basis, self.weights)
        return out
