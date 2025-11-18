"""
Numerical Analysis Module for MathGPU_Kernel_Lab.
Provides precision-aware numerical error analysis, tolerance validation,
and statistical error tracking between reference and GPU-accelerated tensor kernels.
"""

from .accuracy import (
    relative_error,
    max_absolute_error,
    root_mean_square_error,
    snr_db,
    compute_error_metrics,
    assert_close_custom,
)

__all__ = [
    "relative_error",
    "max_absolute_error",
    "root_mean_square_error",
    "snr_db",
    "compute_error_metrics",
    "assert_close_custom",
]
