"""
Numerical Accuracy & Precision Validation Tools.
Calculates numerical distance, floating point cancellation, and relative errors.
"""

from typing import Dict, Any, Union
import torch
import numpy as np


def to_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Ensure input is a torch.Tensor on CPU or CUDA in float64 for ground-truth comparison."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def relative_error(ref: Union[torch.Tensor, np.ndarray],
                   test: Union[torch.Tensor, np.ndarray],
                   eps: float = 1e-12) -> float:
    r"""
    Calculate relative L2 error:
    .. math::
        E_{\text{rel}} = \frac{\|y - \hat{y}\|_2}{\|y\|_2 + \epsilon}
    """
    ref_t = to_tensor(ref).to(dtype=torch.float64)
    test_t = to_tensor(test).to(dtype=torch.float64, device=ref_t.device)

    diff_norm = torch.linalg.norm(ref_t - test_t).item()
    ref_norm = torch.linalg.norm(ref_t).item()

    return diff_norm / (ref_norm + eps)


def max_absolute_error(ref: Union[torch.Tensor, np.ndarray],
                       test: Union[torch.Tensor, np.ndarray]) -> float:
    r"""
    Calculate maximum absolute error (L-infinity norm):
    .. math::
        E_{\text{max}} = \max |y - \hat{y}|
    """
    ref_t = to_tensor(ref).to(dtype=torch.float64)
    test_t = to_tensor(test).to(dtype=torch.float64, device=ref_t.device)
    return torch.max(torch.abs(ref_t - test_t)).item()


def root_mean_square_error(ref: Union[torch.Tensor, np.ndarray],
                           test: Union[torch.Tensor, np.ndarray]) -> float:
    r"""
    Calculate Root Mean Square Error (RMSE):
    .. math::
        \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}
    """
    ref_t = to_tensor(ref).to(dtype=torch.float64)
    test_t = to_tensor(test).to(dtype=torch.float64, device=ref_t.device)
    return torch.sqrt(torch.mean((ref_t - test_t) ** 2)).item()


def snr_db(ref: Union[torch.Tensor, np.ndarray],
           test: Union[torch.Tensor, np.ndarray],
           eps: float = 1e-12) -> float:
    r"""
    Calculate Signal-to-Noise Ratio (SNR) in decibels:
    .. math::
        \text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{\sum y^2}{\sum (y - \hat{y})^2 + \epsilon}\right)
    """
    ref_t = to_tensor(ref).to(dtype=torch.float64)
    test_t = to_tensor(test).to(dtype=torch.float64, device=ref_t.device)
    signal_power = torch.sum(ref_t ** 2).item()
    noise_power = torch.sum((ref_t - test_t) ** 2).item()
    return float(10.0 * np.log10((signal_power + eps) / (noise_power + eps)))


def compute_error_metrics(ref: Union[torch.Tensor, np.ndarray],
                          test: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """Compute a consolidated report of error statistics."""
    return {
        "relative_error": relative_error(ref, test),
        "max_absolute_error": max_absolute_error(ref, test),
        "rmse": root_mean_square_error(ref, test),
        "snr_db": snr_db(ref, test),
    }


def assert_close_custom(ref: torch.Tensor,
                        test: torch.Tensor,
                        rtol: float = 1e-4,
                        atol: float = 1e-4,
                        msg: str = "") -> None:
    """Wrapper around torch.testing.assert_close with formatted diagnostics."""
    try:
        torch.testing.assert_close(test, ref, rtol=rtol, atol=atol)
    except AssertionError as e:
        metrics = compute_error_metrics(ref, test)
        detail = (f"\n[Validation Failed] {msg}\n"
                  f"  - Relative Error: {metrics['relative_error']:.6e} (allowed rtol: {rtol:.1e})\n"
                  f"  - Max Abs Error:  {metrics['max_absolute_error']:.6e} (allowed atol: {atol:.1e})\n"
                  f"  - RMSE:           {metrics['rmse']:.6e}\n"
                  f"  - SNR:            {metrics['snr_db']:.2f} dB\n")
        raise AssertionError(detail + str(e)) from e
