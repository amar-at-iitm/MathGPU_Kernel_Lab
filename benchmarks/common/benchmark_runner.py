"""
Benchmark Orchestration, Table Generation, and CSV Export.
"""

from typing import List, Dict, Any, Optional
import os
import csv
import pandas as pd
from tabulate import tabulate
from .metrics import BenchmarkResult


def format_results_table(results: List[BenchmarkResult], tablefmt: str = "github") -> str:
    """Format benchmark results into a clean markdown or ascii table."""
    headers = [
        "Kernel",
        "Shape (MxNxK)",
        "Latency (ms)",
        "Std (ms)",
        "TFLOPS",
        "Bandwidth (GB/s)",
        "Speedup",
        "Rel Error",
    ]
    rows = []
    for r in results:
        err_str = f"{r.relative_error:.2e}" if r.relative_error is not None else "-"
        rows.append([
            r.kernel_name,
            r.matrix_shape,
            f"{r.mean_ms:.3f}",
            f"{r.std_ms:.3f}",
            f"{r.tflops:.2f}",
            f"{r.bandwidth_gbps:.2f}",
            f"{r.speedup_vs_baseline:.2f}x",
            err_str,
        ])
    return tabulate(rows, headers=headers, tablefmt=tablefmt)


def export_results_csv(results: List[BenchmarkResult], filepath: str) -> None:
    """Save benchmark results to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_csv(filepath, index=False)


class BenchmarkSuite:
    """Harness to manage benchmark runs across configurations."""

    def __init__(self, name: str):
        self.name = name
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def print_summary(self) -> None:
        print(f"\n=======================================================")
        print(f" Benchmark Suite: {self.name}")
        print(f"=======================================================\n")
        print(format_results_table(self.results))
        print("\n")

    def save_csv(self, filepath: str) -> None:
        export_results_csv(self.results, filepath)
        print(f"[Info] Benchmark results exported to: {filepath}")
