import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from typing import List
import torch
from pytorch.baselines.gemm import pytorch_gemm_reference, generate_gemm_inputs
from benchmarks.common.timer import benchmark_function
from benchmarks.common.metrics import (
    calc_flops_gemm,
    calc_tflops,
    calc_gemm_bytes,
    calc_bandwidth_gbps,
    BenchmarkResult,
)
from benchmarks.common.benchmark_runner import BenchmarkSuite


def run_gemm_pytorch_benchmark(sizes: List[int],
                               dtype: torch.dtype = torch.float32,
                               warmup: int = 15,
                               rep: int = 50) -> BenchmarkSuite:
    suite = BenchmarkSuite(name=f"PyTorch cuBLAS Baseline ({str(dtype).replace('torch.', '')})")

    for n in sizes:
        m, k = n, n
        a, b = generate_gemm_inputs(m, n, k, dtype=dtype, device="cuda")

        # Measure latency
        stats = benchmark_function(pytorch_gemm_reference, a, b, warmup=warmup, rep=rep)

        flops = calc_flops_gemm(m, n, k)
        tflops = calc_tflops(flops, stats["mean_ms"])

        bytes_count = calc_gemm_bytes(m, n, k, bytes_per_element=a.element_size())
        bw = calc_bandwidth_gbps(bytes_count, stats["mean_ms"])

        res = BenchmarkResult(
            kernel_name=f"PyTorch ({str(dtype).replace('torch.', '')})",
            matrix_shape=f"{m}x{n}x{k}",
            mean_ms=stats["mean_ms"],
            std_ms=stats["std_ms"],
            tflops=tflops,
            bandwidth_gbps=bw,
            speedup_vs_baseline=1.0,
            relative_error=0.0,
        )
        suite.add_result(res)

    return suite


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch cuBLAS GEMM")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048], help="Matrix dimensions")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--out-csv", type=str, default="results/benchmarks/gemm_pytorch_baseline.csv")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    suite = run_gemm_pytorch_benchmark(args.sizes, dtype=dtype)
    suite.print_summary()
    suite.save_csv(args.out_csv)


if __name__ == "__main__":
    main()
