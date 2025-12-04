"""
Performance Metrics & Theoretical Throughput Calculators.
Computes TFLOPS, Memory Bandwidth (GB/s), and Relative Speedup.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class BenchmarkResult:
    kernel_name: str
    matrix_shape: str
    mean_ms: float
    std_ms: float
    tflops: float
    bandwidth_gbps: float
    speedup_vs_baseline: float = 1.0
    relative_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calc_flops_gemm(m: int, n: int, k: int) -> int:
    r"""GEMM operation count: C = A * B requires 2 * M * N * K FLOPs."""
    return 2 * m * n * k


def calc_tflops(flops: float, time_ms: float) -> float:
    r"""
    Calculate Floating Point Operations per Second in TeraFLOPS (10^12 FLOPS).
    time_ms is in milliseconds.
    """
    if time_ms <= 0:
        return 0.0
    return (flops / (time_ms * 1e-3)) / 1e12


def calc_bandwidth_gbps(bytes_transferred: float, time_ms: float) -> float:
    r"""
    Calculate effective memory throughput in Gigabytes per second (GB/s = 10^9 Bytes/s).
    """
    if time_ms <= 0:
        return 0.0
    return (bytes_transferred / (time_ms * 1e-3)) / 1e9


def calc_gemm_bytes(m: int, n: int, k: int, bytes_per_element: int = 4) -> int:
    r"""
    Theoretical minimum DRAM traffic for GEMM:
    Read A (M * K), Read B (K * N), Write C (M * N).
    """
    elements = (m * k) + (k * n) + (m * n)
    return elements * bytes_per_element
