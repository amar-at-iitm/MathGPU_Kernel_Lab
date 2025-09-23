"""
PyTorch / cuBLAS Reference Implementation of General Matrix Multiplication (GEMM).
Mathematical operation: C = alpha * (A @ B) + beta * C
"""

from typing import Optional, Tuple
import torch
from numerical_analysis.accuracy import assert_close_custom, compute_error_metrics


def pytorch_gemm_reference(a: torch.Tensor,
                           b: torch.Tensor,
                           alpha: float = 1.0,
                           beta: float = 0.0,
                           c: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute GEMM: C = alpha * (A @ B) + beta * C using PyTorch's native BLAS (cuBLAS).

    Args:
        a: Matrix A of shape (M, K)
        b: Matrix B of shape (K, N)
        alpha: Scalar multiplier for AB
        beta: Scalar multiplier for C
        c: Optional existing matrix C of shape (M, N)

    Returns:
        Result matrix C of shape (M, N)
    """
    assert a.dim() == 2 and b.dim() == 2, "A and B must be 2D matrices"
    assert a.size(1) == b.size(0), f"Dimension mismatch: A({a.size(0)}, {a.size(1)}) and B({b.size(0)}, {b.size(1)})"

    if alpha == 1.0 and beta == 0.0:
        return torch.matmul(a, b)

    out = alpha * torch.matmul(a, b)
    if beta != 0.0 and c is not None:
        out = out + beta * c
    return out


def generate_gemm_inputs(m: int,
                         n: int,
                         k: int,
                         dtype: torch.dtype = torch.float32,
                         device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate normalized random inputs for reproducible GEMM benchmarking."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    # Normalize inputs to prevent extreme overflow on large accumulations
    scale = 1.0 / (k ** 0.5)
    a = torch.randn(m, k, dtype=dtype, device=dev) * scale
    b = torch.randn(k, n, dtype=dtype, device=dev) * scale
    return a, b


def verify_gemm(test_c: torch.Tensor,
                a: torch.Tensor,
                b: torch.Tensor,
                alpha: float = 1.0,
                beta: float = 0.0,
                c: Optional[torch.Tensor] = None,
                rtol: float = 1e-4,
                atol: float = 1e-4) -> dict:
    """Validate kernel output against high-precision float64 ground truth."""
    # Compute ground truth in FP64
    a_f64 = a.to(dtype=torch.float64)
    b_f64 = b.to(dtype=torch.float64)
    c_f64 = c.to(dtype=torch.float64) if c is not None else None

    ref = pytorch_gemm_reference(a_f64, b_f64, alpha=alpha, beta=beta, c=c_f64)
    metrics = compute_error_metrics(ref, test_c)
    assert_close_custom(ref.to(dtype=test_c.dtype), test_c, rtol=rtol, atol=atol, msg="GEMM Verification")
    return metrics
