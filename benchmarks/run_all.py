"""
End-to-End Master Benchmark Suite.
Benchmarks all 5 core mathematical domains across CUDA, Triton, and PyTorch baselines:
1. Matrix Multiplication (GEMM)
2. Gaussian RBF & KAN Layer
3. Numerically Stable Softmax
4. Scaled Dot-Product FlashAttention
5. 4D Tensor Contraction
Exports consolidated results to results/benchmarks/end_to_end_benchmark_summary.csv.
"""

import os
import sys
from pathlib import Path
import csv
import torch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common.timer import benchmark_function
from numerical_analysis.accuracy import relative_error

# 1. GEMM
from pytorch.baselines.gemm import pytorch_gemm_reference
from cuda.naive.gemm_naive_runner import cuda_gemm_naive
from cuda.tiled.gemm_tiled_runner import cuda_gemm_tiled
from cuda.optimized.gemm_reg_blocked_runner import cuda_gemm_reg_blocked
from cuda.tensor_core.gemm_wmma_runner import cuda_gemm_wmma
from triton_kernels.gemm import triton_gemm

# 2. RBF & KAN
from pytorch.baselines.rbf import rbf_kernel_pytorch
from cuda.optimized.rbf_kernel_runner import cuda_rbf_kernel_fused
from triton_kernels.rbf import triton_rbf_kernel
from pytorch.baselines.kan_layer import PyTorchKANRBFLayer
from cuda.optimized.kan_layer_runner import cuda_kan_rbf_forward
from triton_kernels.kan import triton_kan_rbf_forward

# 3. Softmax
from pytorch.baselines.softmax import pytorch_softmax_reference
from cuda.optimized.softmax_warp_runner import cuda_softmax_warp
from triton_kernels.softmax import triton_softmax

# 4. Attention
from pytorch.baselines.attention import scaled_dot_product_attention_pytorch
from cuda.naive.attention_naive_runner import cuda_attention_naive
from cuda.optimized.attention_fused_runner import cuda_attention_fused
from triton_kernels.attention import triton_flash_attention

# 5. Tensor Contraction
from cuda.optimized.tensor_contraction_runner import cuda_tensor_contraction_4d
from triton_kernels.tensor_contraction import triton_tensor_contraction_4d


def run_all_benchmarks(warmup=5, repeat=15):
    assert torch.cuda.is_available(), "CUDA required for benchmarks"
    results = []

    print("================================================================================")
    print("           STARTING MATHGPU KERNEL LAB END-TO-END BENCHMARK SUITE")
    print(f"  Target Device: {torch.cuda.get_device_name(0)}")
    print("================================================================================")

    # -------------------------------------------------------------
    # Domain 1: GEMM (M = N = K = 1024)
    # -------------------------------------------------------------
    print("\n[1/5] Benchmarking Matrix Multiplication (M=N=K=1024)...")
    m_sz = 1024
    ga = torch.randn(m_sz, m_sz, device="cuda", dtype=torch.float32)
    gb = torch.randn(m_sz, m_sz, device="cuda", dtype=torch.float32)
    ga_fp16 = ga.half()
    gb_fp16 = gb.half()
    gemm_flops = 2.0 * m_sz * m_sz * m_sz
    g_ref = pytorch_gemm_reference(ga, gb)

    gemm_benchmarks = [
        ("PyTorch cuBLAS", lambda: pytorch_gemm_reference(ga, gb)),
        ("Naive CUDA", lambda: cuda_gemm_naive(ga, gb)),
        ("Tiled CUDA (SRAM)", lambda: cuda_gemm_tiled(ga, gb)),
        ("Register Blocked CUDA", lambda: cuda_gemm_reg_blocked(ga, gb)),
        ("Tensor Core WMMA (FP16)", lambda: cuda_gemm_wmma(ga_fp16, gb_fp16)),
        ("Triton GEMM", lambda: triton_gemm(ga, gb)),
    ]

    base_g_ms = None
    for name, fn in gemm_benchmarks:
        out = fn()
        rel_err = relative_error(g_ref, out)
        stats = benchmark_function(fn, warmup=warmup, rep=repeat)
        ms = stats["mean_ms"]
        if base_g_ms is None:
            base_g_ms = ms
        tflops = (gemm_flops / (ms * 1e-3)) / 1e12
        results.append({
            "domain": "GEMM",
            "kernel": name,
            "problem_size": f"{m_sz}x{m_sz}",
            "mean_ms": ms,
            "std_ms": stats["std_ms"],
            "throughput_metric": f"{tflops:.2f} TFLOPS",
            "speedup_vs_baseline": base_g_ms / ms,
            "relative_error": rel_err,
        })

    # -------------------------------------------------------------
    # Domain 2: Gaussian RBF & KAN (M=2048, N=2048, D=64)
    # -------------------------------------------------------------
    print("\n[2/5] Benchmarking Gaussian RBF & KAN Layer...")
    rx = torch.randn(2048, 64, device="cuda", dtype=torch.float32)
    ry = torch.randn(2048, 64, device="cuda", dtype=torch.float32)
    r_ref = rbf_kernel_pytorch(rx, ry, sigma=1.0)

    rbf_benchmarks = [
        ("PyTorch RBF Baseline", lambda: rbf_kernel_pytorch(rx, ry, sigma=1.0)),
        ("Fused CUDA RBF", lambda: cuda_rbf_kernel_fused(rx, ry, sigma=1.0)),
        ("Triton RBF Kernel", lambda: triton_rbf_kernel(rx, ry, sigma=1.0)),
    ]

    base_r_ms = None
    for name, fn in rbf_benchmarks:
        out = fn()
        rel_err = relative_error(r_ref, out)
        stats = benchmark_function(fn, warmup=warmup, rep=repeat)
        ms = stats["mean_ms"]
        if base_r_ms is None:
            base_r_ms = ms
        results.append({
            "domain": "Gaussian RBF",
            "kernel": name,
            "problem_size": "2048x2048 (D=64)",
            "mean_ms": ms,
            "std_ms": stats["std_ms"],
            "throughput_metric": f"{(2048*2048*64*2 / (ms*1e-3)) / 1e9:.2f} GFLOPS",
            "speedup_vs_baseline": base_r_ms / ms,
            "relative_error": rel_err,
        })

    # KAN Layer benchmark (B=1024, In=64, Out=64, G=8)
    kan_x = torch.randn(1024, 64, device="cuda", dtype=torch.float32)
    kan_py = PyTorchKANRBFLayer(in_features=64, out_features=64, num_bases=8).cuda()
    kan_ref = kan_py(kan_x)

    kan_benchmarks = [
        ("PyTorch KAN Layer", lambda: kan_py(kan_x)),
        ("CUDA Fused KAN", lambda: cuda_kan_rbf_forward(kan_x, kan_py.weights, kan_py.grid_mu, kan_py.grid_sigma)),
        ("Triton KAN Layer", lambda: triton_kan_rbf_forward(kan_x, kan_py.weights, kan_py.grid_mu, kan_py.grid_sigma)),
    ]
    base_k_ms = None
    for name, fn in kan_benchmarks:
        out = fn()
        rel_err = relative_error(kan_ref, out)
        stats = benchmark_function(fn, warmup=warmup, rep=repeat)
        ms = stats["mean_ms"]
        if base_k_ms is None:
            base_k_ms = ms
        results.append({
            "domain": "KAN Layer",
            "kernel": name,
            "problem_size": "1024x64 (G=8, Out=64)",
            "mean_ms": ms,
            "std_ms": stats["std_ms"],
            "throughput_metric": f"{(1024*64*8*64*2 / (ms*1e-3)) / 1e9:.2f} GFLOPS",
            "speedup_vs_baseline": base_k_ms / ms,
            "relative_error": rel_err,
        })

    # -------------------------------------------------------------
    # Domain 3: Softmax (Batch=4096, Dim=4096)
    # -------------------------------------------------------------
    print("\n[3/5] Benchmarking Softmax (Batch=4096, Dim=4096)...")
    sm_x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    sm_ref = pytorch_softmax_reference(sm_x, dim=-1)

    sm_benchmarks = [
        ("PyTorch Softmax", lambda: pytorch_softmax_reference(sm_x, dim=-1)),
        ("Warp Shuffle CUDA", lambda: cuda_softmax_warp(sm_x)),
        ("Triton Softmax", lambda: triton_softmax(sm_x)),
    ]

    base_sm_ms = None
    for name, fn in sm_benchmarks:
        out = fn()
        rel_err = relative_error(sm_ref, out)
        stats = benchmark_function(fn, warmup=warmup, rep=repeat)
        ms = stats["mean_ms"]
        if base_sm_ms is None:
            base_sm_ms = ms
        results.append({
            "domain": "Softmax",
            "kernel": name,
            "problem_size": "4096x4096",
            "mean_ms": ms,
            "std_ms": stats["std_ms"],
            "throughput_metric": f"{(4096*4096*4*2 / (ms*1e-3))/1e9:.2f} GB/s",
            "speedup_vs_baseline": base_sm_ms / ms,
            "relative_error": rel_err,
        })

    # -------------------------------------------------------------
    # Domain 4: Attention (Batch=8, Seq=1024, Dim=64)
    # -------------------------------------------------------------
    print("\n[4/5] Benchmarking Attention (B=8, Seq=1024, D=64)...")
    B, S, D = 8, 1024, 64
    aq = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    ak = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    av = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    att_ref = scaled_dot_product_attention_pytorch(aq, ak, av)
    att_flops = 4.0 * B * S * S * D

    att_benchmarks = [
        ("PyTorch SDPA", lambda: scaled_dot_product_attention_pytorch(aq, ak, av)),
        ("Naive CUDA Attention", lambda: cuda_attention_naive(aq, ak, av)),
        ("Fused FlashAttention CUDA", lambda: cuda_attention_fused(aq, ak, av)),
        ("Triton FlashAttention", lambda: triton_flash_attention(aq, ak, av)),
    ]

    base_att_ms = None
    for name, fn in att_benchmarks:
        out = fn()
        rel_err = relative_error(att_ref, out)
        stats = benchmark_function(fn, warmup=warmup, rep=repeat)
        ms = stats["mean_ms"]
        if base_att_ms is None:
            base_att_ms = ms
        tflops = (att_flops / (ms * 1e-3)) / 1e12
        results.append({
            "domain": "Attention",
            "kernel": name,
            "problem_size": f"B={B},S={S},D={D}",
            "mean_ms": ms,
            "std_ms": stats["std_ms"],
            "throughput_metric": f"{tflops:.2f} TFLOPS",
            "speedup_vs_baseline": base_att_ms / ms,
            "relative_error": rel_err,
        })

    # -------------------------------------------------------------
    # Domain 5: 4D Tensor Contraction (I=8, J=16, K=8, L=16, A=16, B=16)
    # -------------------------------------------------------------
    print("\n[5/5] Benchmarking 4D Tensor Contraction...")
    I, J, K_dim, L = 8, 16, 8, 16
    A_dim, B_dim = 16, 16
    scale = 1.0 / ((A_dim * B_dim) ** 0.5)
    tc_A = torch.randn(I, J, A_dim, B_dim, device="cuda", dtype=torch.float32) * scale
    tc_B = torch.randn(A_dim, B_dim, K_dim, L, device="cuda", dtype=torch.float32) * scale
    tc_ref = torch.einsum("ijab,abkl->ijkl", tc_A, tc_B)
    tc_flops = 2.0 * I * J * K_dim * L * A_dim * B_dim

    tc_benchmarks = [
        ("PyTorch einsum", lambda: torch.einsum("ijab,abkl->ijkl", tc_A, tc_B)),
        ("Tiled CUDA Contraction", lambda: cuda_tensor_contraction_4d(tc_A, tc_B)),
        ("Triton Contraction", lambda: triton_tensor_contraction_4d(tc_A, tc_B)),
    ]

    base_tc_ms = None
    for name, fn in tc_benchmarks:
        out = fn()
        rel_err = relative_error(tc_ref, out)
        stats = benchmark_function(fn, warmup=warmup, rep=repeat)
        ms = stats["mean_ms"]
        if base_tc_ms is None:
            base_tc_ms = ms
        tflops = (tc_flops / (ms * 1e-3)) / 1e12
        results.append({
            "domain": "Tensor Contraction",
            "kernel": name,
            "problem_size": f"{I}x{J}x{K_dim}x{L}",
            "mean_ms": ms,
            "std_ms": stats["std_ms"],
            "throughput_metric": f"{tflops:.2f} TFLOPS",
            "speedup_vs_baseline": base_tc_ms / ms,
            "relative_error": rel_err,
        })

    # Save to CSV
    csv_path = "results/benchmarks/end_to_end_benchmark_summary.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary table
    table_rows = [
        [r["domain"], r["kernel"], r["problem_size"], f"{r['mean_ms']:.4f} ms", r["throughput_metric"], f"{r['speedup_vs_baseline']:.2f}x", f"{r['relative_error']:.2e}"]
        for r in results
    ]
    print("\n" + "=" * 105)
    print("                              MASTER BENCHMARK SUMMARY REPORT")
    print("=" * 105)
    print(tabulate(table_rows, headers=["Domain", "Kernel Implementation", "Problem Size", "Latency", "Throughput", "Speedup", "Rel Err"], tablefmt="github"))
    print("=" * 105)
    print(f"\nBenchmark summary saved to: {csv_path}\n")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
