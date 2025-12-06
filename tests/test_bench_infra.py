"""
Tests for benchmark timer, metrics, and suite formatting.
"""

import pytest
import torch
from benchmarks.common.timer import CudaTimer, benchmark_function
from benchmarks.common.metrics import (
    calc_flops_gemm,
    calc_tflops,
    calc_bandwidth_gbps,
    BenchmarkResult,
)
from benchmarks.common.benchmark_runner import format_results_table


def test_cuda_timer_context():
    """Verify CudaTimer context manager records positive elapsed time."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")

    with CudaTimer() as timer:
        c = torch.matmul(a, b)

    assert timer.elapsed_ms > 0.0
    print(f"\n[Test Info] 512x512 GEMM time via CudaTimer: {timer.elapsed_ms:.4f} ms")


def test_benchmark_function_statistics():
    """Verify benchmark_function returns valid statistical distribution."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")

    stats = benchmark_function(torch.matmul, a, b, warmup=5, rep=20)
    assert "mean_ms" in stats
    assert "median_ms" in stats
    assert "std_ms" in stats
    assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]
    assert stats["mean_ms"] > 0.0


def test_metrics_calculations():
    """Verify FLOPs and TFLOPS math."""
    flops = calc_flops_gemm(1024, 1024, 1024)
    expected_flops = 2 * 1024 * 1024 * 1024
    assert flops == expected_flops

    # If 2 GFLOPS runs in 1.0 ms -> 2 TFLOPS
    tflops = calc_tflops(2e9, 1.0)
    assert abs(tflops - 2.0) < 1e-4

    # 1 GB in 1 ms -> 1000 GB/s
    bw = calc_bandwidth_gbps(1e6, 1.0)
    assert abs(bw - 1.0) < 1e-4


def test_format_results_table():
    """Verify format_results_table creates valid table string."""
    res = [
        BenchmarkResult(
            kernel_name="PyTorch cuBLAS",
            matrix_shape="1024x1024x1024",
            mean_ms=0.5,
            std_ms=0.02,
            tflops=4.29,
            bandwidth_gbps=25.1,
            speedup_vs_baseline=1.0,
            relative_error=0.0,
        )
    ]
    table = format_results_table(res)
    assert "PyTorch cuBLAS" in table
    assert "1024x1024x1024" in table
