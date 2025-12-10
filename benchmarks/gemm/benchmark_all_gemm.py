"""
Master GEMM Benchmark Suite:
Compares cuBLAS vs. Naive CUDA vs. Tiled Shared Mem vs. Register Blocked vs. Tensor Core WMMA vs. Triton.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import torch
from pytorch.baselines.gemm import pytorch_gemm_reference, generate_gemm_inputs
from cuda.naive.gemm_naive_runner import cuda_gemm_naive
from cuda.tiled.gemm_tiled_runner import cuda_gemm_tiled
from cuda.optimized.gemm_reg_blocked_runner import cuda_gemm_reg_blocked
from cuda.tensor_core.gemm_wmma_runner import cuda_gemm_wmma
from triton_kernels.gemm import triton_gemm
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


def run_all_gemm_benchmark(sizes=[256, 512, 1024], warmup=10, rep=30) -> BenchmarkSuite:
    suite = BenchmarkSuite(name="Comprehensive GEMM Benchmark (All Architectures & Kernels)")

    for n in sizes:
        m, k = n, n
        a, b = generate_gemm_inputs(m, n, k, dtype=torch.float32, device="cuda")
        a_fp16 = a.half()
        b_fp16 = b.half()
        flops = calc_flops_gemm(m, n, k)

        # 1. PyTorch cuBLAS
        ref_out = pytorch_gemm_reference(a, b)
        stats = benchmark_function(pytorch_gemm_reference, a, b, warmup=warmup, rep=rep)
        baseline_ms = stats["mean_ms"]
        suite.add_result(BenchmarkResult(
            kernel_name="PyTorch cuBLAS",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats["mean_ms"],
            std_ms=stats["std_ms"],
            tflops=calc_tflops(flops, stats["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(calc_gemm_bytes(m, n, k, 4), stats["mean_ms"]),
            speedup_vs_baseline=1.0,
            relative_error=0.0,
        ))

        # 2. Naive CUDA
        naive_out = cuda_gemm_naive(a, b)
        stats = benchmark_function(cuda_gemm_naive, a, b, warmup=warmup, rep=rep)
        suite.add_result(BenchmarkResult(
            kernel_name="Naive CUDA",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats["mean_ms"],
            std_ms=stats["std_ms"],
            tflops=calc_tflops(flops, stats["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(calc_gemm_bytes(m, n, k, 4), stats["mean_ms"]),
            speedup_vs_baseline=baseline_ms / stats["mean_ms"],
            relative_error=relative_error(ref_out, naive_out),
        ))

        # 3. Tiled CUDA (Shared Memory)
        tiled_out = cuda_gemm_tiled(a, b)
        stats = benchmark_function(cuda_gemm_tiled, a, b, warmup=warmup, rep=rep)
        suite.add_result(BenchmarkResult(
            kernel_name="Tiled CUDA (SRAM)",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats["mean_ms"],
            std_ms=stats["std_ms"],
            tflops=calc_tflops(flops, stats["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(calc_gemm_bytes(m, n, k, 4), stats["mean_ms"]),
            speedup_vs_baseline=baseline_ms / stats["mean_ms"],
            relative_error=relative_error(ref_out, tiled_out),
        ))

        # 4. Register-Blocked CUDA
        reg_out = cuda_gemm_reg_blocked(a, b)
        stats = benchmark_function(cuda_gemm_reg_blocked, a, b, warmup=warmup, rep=rep)
        suite.add_result(BenchmarkResult(
            kernel_name="Register Blocked CUDA",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats["mean_ms"],
            std_ms=stats["std_ms"],
            tflops=calc_tflops(flops, stats["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(calc_gemm_bytes(m, n, k, 4), stats["mean_ms"]),
            speedup_vs_baseline=baseline_ms / stats["mean_ms"],
            relative_error=relative_error(ref_out, reg_out),
        ))

        # 5. Tensor Core WMMA FP16
        wmma_out = cuda_gemm_wmma(a_fp16, b_fp16)
        stats = benchmark_function(cuda_gemm_wmma, a_fp16, b_fp16, warmup=warmup, rep=rep)
        suite.add_result(BenchmarkResult(
            kernel_name="Tensor Core WMMA (FP16)",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats["mean_ms"],
            std_ms=stats["std_ms"],
            tflops=calc_tflops(flops, stats["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(calc_gemm_bytes(m, n, k, 2), stats["mean_ms"]),
            speedup_vs_baseline=baseline_ms / stats["mean_ms"],
            relative_error=relative_error(ref_out, wmma_out),
        ))

        # 6. Triton GEMM
        triton_out = triton_gemm(a, b)
        stats = benchmark_function(triton_gemm, a, b, warmup=warmup, rep=rep)
        suite.add_result(BenchmarkResult(
            kernel_name="Triton JIT GEMM",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats["mean_ms"],
            std_ms=stats["std_ms"],
            tflops=calc_tflops(flops, stats["mean_ms"]),
            bandwidth_gbps=calc_bandwidth_gbps(calc_gemm_bytes(m, n, k, 4), stats["mean_ms"]),
            speedup_vs_baseline=baseline_ms / stats["mean_ms"],
            relative_error=relative_error(ref_out, triton_out),
        ))

    return suite


def main():
    parser = argparse.ArgumentParser(description="Run Master GEMM Benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024], help="Matrix dimensions")
    parser.add_argument("--out-csv", type=str, default="results/benchmarks/all_gemm_comparison.csv")
    args = parser.parse_args()

    suite = run_all_gemm_benchmark(args.sizes)
    suite.print_summary()
    suite.save_csv(args.out_csv)


if __name__ == "__main__":
    main()
