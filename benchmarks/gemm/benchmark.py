"""
Comparative GEMM Benchmark:
PyTorch cuBLAS vs. Naive CUDA GEMM vs. Tiled Shared-Memory CUDA GEMM.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from typing import List
import torch
from pytorch.baselines.gemm import pytorch_gemm_reference, generate_gemm_inputs
from cuda.naive.gemm_naive_runner import cuda_gemm_naive
from cuda.tiled.gemm_tiled_runner import cuda_gemm_tiled
from numerical_analysis.accuracy import relative_error
from benchmarks.common.timer import benchmark_function
from benchmarks.common.metrics import (
    calc_flops_gemm,
    calc_tflops,
    calc_gemm_bytes,
    calc_bandwidth_gbps,
    BenchmarkResult,
)
from benchmarks.common.benchmark_runner import BenchmarkSuite


def run_comparative_gemm_benchmark(sizes: List[int],
                                   warmup: int = 10,
                                   rep: int = 30) -> BenchmarkSuite:
    suite = BenchmarkSuite(name="GEMM Comparative Benchmark (cuBLAS vs. Naive CUDA vs. Tiled CUDA)")

    for n in sizes:
        m, k = n, n
        a, b = generate_gemm_inputs(m, n, k, dtype=torch.float32, device="cuda")
        flops = calc_flops_gemm(m, n, k)
        bytes_count = calc_gemm_bytes(m, n, k, bytes_per_element=4)

        # 1. Reference: PyTorch cuBLAS
        ref_out = pytorch_gemm_reference(a, b)
        stats_cublas = benchmark_function(pytorch_gemm_reference, a, b, warmup=warmup, rep=rep)
        res_cublas = BenchmarkResult(
            kernel_name="PyTorch cuBLAS",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats_cublas["mean_ms"],
            std_ms=stats_cublas["std_ms"],
            tflops=calc_tflops(flops, stats_cublas["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(bytes_count, stats_cublas["mean_ms"]),
            speedup_vs_baseline=1.0,
            relative_error=0.0,
        )
        suite.add_result(res_cublas)
        baseline_ms = stats_cublas["mean_ms"]

        # 2. Naive CUDA GEMM
        naive_out = cuda_gemm_naive(a, b)
        err_naive = relative_error(ref_out, naive_out)
        stats_naive = benchmark_function(cuda_gemm_naive, a, b, warmup=warmup, rep=rep)
        res_naive = BenchmarkResult(
            kernel_name="Naive CUDA",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats_naive["mean_ms"],
            std_ms=stats_naive["std_ms"],
            tflops=calc_tflops(flops, stats_naive["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(bytes_count, stats_naive["mean_ms"]),
            speedup_vs_baseline=baseline_ms / stats_naive["mean_ms"],
            relative_error=err_naive,
        )
        suite.add_result(res_naive)

        # 3. Tiled Shared-Memory CUDA GEMM
        tiled_out = cuda_gemm_tiled(a, b)
        err_tiled = relative_error(ref_out, tiled_out)
        stats_tiled = benchmark_function(cuda_gemm_tiled, a, b, warmup=warmup, rep=rep)
        res_tiled = BenchmarkResult(
            kernel_name="Tiled CUDA (Shared Mem)",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats_tiled["mean_ms"],
            std_ms=stats_tiled["std_ms"],
            tflops=calc_tflops(flops, stats_tiled["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(bytes_count, stats_tiled["mean_ms"]),
            speedup_vs_baseline=baseline_ms / stats_tiled["mean_ms"],
            relative_error=err_tiled,
        )
        suite.add_result(res_tiled)

    return suite


def main():
    parser = argparse.ArgumentParser(description="Run comparative GEMM benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024], help="Matrix dimensions to test")
    parser.add_argument("--out-csv", type=str, default="results/benchmarks/gemm_comparison.csv", help="Output CSV path")
    args = parser.parse_args()

    suite = run_comparative_gemm_benchmark(args.sizes)
    suite.print_summary()
    suite.save_csv(args.out_csv)


if __name__ == "__main__":
    main()
