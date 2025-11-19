"""
Mixed-Precision Accuracy & Floating-Point Error Analysis Suite.
Quantifies precision loss across FP64 (Ground Truth), FP32 (strict IEEE 754),
TF32 (TensorFloat-32), FP16 (Half), and BF16 (BFloat16) across matrix conditioning numbers.
"""

import os
import sys
from pathlib import Path
import csv
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from numerical_analysis.accuracy import compute_error_metrics


def generate_conditioned_matrix(n: int,
                                condition_number: float,
                                device: str = "cuda") -> torch.Tensor:
    """
    Generate an n x n matrix with an exact prescribed condition number using SVD synthesis:
    A = U * S * V^T, where S has singular values geometrically spaced between 1.0 and 1/condition_number.
    """
    # Create random orthogonal matrices U and V via QR factorization in FP64
    x = torch.randn(n, n, dtype=torch.float64, device="cpu")
    q_u, _ = torch.linalg.qr(x)

    y = torch.randn(n, n, dtype=torch.float64, device="cpu")
    q_v, _ = torch.linalg.qr(y)

    # Singular values: exponentially distributed from 1.0 down to 1.0 / condition_number
    s = torch.logspace(0, -np.log10(condition_number), steps=n, dtype=torch.float64, device="cpu")
    sigma = torch.diag(s)

    a_fp64 = (q_u @ sigma @ q_v.T).to(device=device)
    return a_fp64


def evaluate_precision_modes(n: int = 512,
                             condition_number: float = 100.0,
                             device: str = "cuda") -> List[Dict[str, any]]:
    """
    Evaluate GEMM accuracy for FP32 (strict), TF32, FP16, and BF16 against FP64 ground truth.
    """
    assert torch.cuda.is_available(), "CUDA required for precision evaluation"

    # 1. Ground truth in FP64
    A_64 = generate_conditioned_matrix(n, condition_number, device=device)
    B_64 = generate_conditioned_matrix(n, condition_number, device=device)
    Y_ref = A_64 @ B_64

    results = []

    # 2. FP32 Strict (IEEE 754 - TF32 disabled)
    orig_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        A_32 = A_64.to(torch.float32)
        B_32 = B_64.to(torch.float32)
        Y_fp32 = A_32 @ B_32
        metrics_fp32 = compute_error_metrics(Y_ref, Y_fp32)
        results.append({
            "size": n,
            "condition_number": condition_number,
            "precision": "FP32 (IEEE)",
            **metrics_fp32
        })

        # 3. TF32 (TensorFloat-32 on Tensor Cores)
        torch.backends.cuda.matmul.allow_tf32 = True
        Y_tf32 = A_32 @ B_32
        metrics_tf32 = compute_error_metrics(Y_ref, Y_tf32)
        results.append({
            "size": n,
            "condition_number": condition_number,
            "precision": "TF32",
            **metrics_tf32
        })
    finally:
        torch.backends.cuda.matmul.allow_tf32 = orig_tf32

    # 4. FP16 (Half precision)
    A_16 = A_64.to(torch.float16)
    B_16 = B_64.to(torch.float16)
    Y_fp16 = A_16 @ B_16
    metrics_fp16 = compute_error_metrics(Y_ref, Y_fp16)
    results.append({
        "size": n,
        "condition_number": condition_number,
        "precision": "FP16",
        **metrics_fp16
    })

    # 5. BF16 (BFloat16)
    A_bf16 = A_64.to(torch.bfloat16)
    B_bf16 = B_64.to(torch.bfloat16)
    Y_bf16 = A_bf16 @ B_bf16
    metrics_bf16 = compute_error_metrics(Y_ref, Y_bf16)
    results.append({
        "size": n,
        "condition_number": condition_number,
        "precision": "BF16",
        **metrics_bf16
    })

    return results


def run_precision_benchmark(sizes: List[int] = [256, 512],
                            condition_numbers: List[float] = [1.0, 10.0, 100.0, 1000.0],
                            output_csv: str = "results/benchmarks/precision_error_report.csv") -> List[Dict[str, any]]:
    """Execute full precision benchmark suite across matrix dimensions and condition numbers."""
    all_rows = []
    print(f"Running Precision & Conditioning Benchmark across sizes={sizes}, cond={condition_numbers}...")

    for sz in sizes:
        for cond in condition_numbers:
            rows = evaluate_precision_modes(n=sz, condition_number=cond)
            all_rows.extend(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved precision report with {len(all_rows)} evaluations to: {output_csv}")

    return all_rows


if __name__ == "__main__":
    import sys
    from tabulate import tabulate
    rows = run_precision_benchmark(sizes=[256, 512], condition_numbers=[1.0, 100.0, 10000.0])
    table_data = [[r["size"], r["condition_number"], r["precision"], f"{r['relative_error']:.2e}", f"{r['max_absolute_error']:.2e}", f"{r['snr_db']:.1f}"] for r in rows]
    print("\n" + tabulate(table_data, headers=["Size", "Cond Num", "Precision", "Rel Error", "Max Abs Err", "SNR (dB)"], tablefmt="github"))
