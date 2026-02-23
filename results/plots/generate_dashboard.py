"""
Performance Dashboard & Visualization Plot Generator.
Parses benchmark and numerical analysis CSVs to produce publication-grade charts.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration for sleek, modern scientific aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_gemm_throughput(summary_rows):
    """Plot GEMM latency and TFLOPS across implementations."""
    gemm_data = [r for r in summary_rows if r["domain"] == "GEMM"]
    if not gemm_data:
        return

    kernels = [r["kernel"] for r in gemm_data]
    latencies = [float(r["mean_ms"]) for r in gemm_data]
    tflops = [float(r["throughput_metric"].split()[0]) for r in gemm_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#4A90E2", "#90A4AE", "#78909C", "#5C6BC0", "#AB47BC", "#00C853"]

    # TFLOPS
    bars1 = ax1.bar(kernels, tflops, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax1.set_title("GEMM Compute Throughput (M=N=K=1024)")
    ax1.set_ylabel("Throughput (TFLOPS)")
    ax1.tick_params(axis="x", rotation=30)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.25, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Latency
    bars2 = ax2.bar(kernels, latencies, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax2.set_title("GEMM Kernel Latency (Lower is Better)")
    ax2.set_ylabel("Execution Time (ms)")
    ax2.tick_params(axis="x", rotation=30)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.2f}ms", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gemm_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Generated: {path}")


def plot_speedups(summary_rows):
    """Plot speedup vs baseline across all kernel domains."""
    labels = []
    speedups = []
    colors = []

    palette = {
        "GEMM": "#00C853",
        "Gaussian RBF": "#29B6F6",
        "KAN Layer": "#AB47BC",
        "Softmax": "#FFA726",
        "Attention": "#FF5252",
        "Tensor Contraction": "#7E57C2",
    }

    for r in summary_rows:
        sp = float(r["speedup_vs_baseline"])
        labels.append(f"{r['domain']}: {r['kernel']}")
        speedups.append(sp)
        colors.append(palette.get(r["domain"], "#78909C"))

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, speedups, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="PyTorch Baseline (1.0x)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Speedup vs PyTorch Baseline (x)")
    ax.set_title("Speedup Factor across CUDA & Triton Implementations")
    ax.legend(loc="lower right")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}x", ha="left", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "kernel_speedups.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Generated: {path}")


def plot_precision_tradeoffs(precision_csv="results/benchmarks/precision_error_report.csv"):
    """Plot Signal-to-Noise Ratio (dB) vs Condition Number."""
    if not os.path.exists(precision_csv):
        return

    with open(precision_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter for size=512
    rows_512 = [r for r in rows if r["size"] == "512"]
    if not rows_512:
        rows_512 = rows

    precisions = sorted(list(set(r["precision"] for r in rows_512)))
    conds = sorted(list(set(float(r["condition_number"]) for r in rows_512)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    color_map = {
        "FP32 (IEEE)": "#2E7D32",
        "TF32": "#1565C0",
        "FP16": "#E65100",
        "BF16": "#C2185B",
    }

    for prec in precisions:
        p_rows = [r for r in rows_512 if r["precision"] == prec]
        p_rows.sort(key=lambda x: float(x["condition_number"]))
        x = [float(r["condition_number"]) for r in p_rows]
        snr = [float(r["snr_db"]) for r in p_rows]
        rel_err = [float(r["relative_error"]) for r in p_rows]

        c = color_map.get(prec, "black")
        ax1.plot(x, snr, marker="o", linewidth=2, label=prec, color=c)
        ax2.plot(x, rel_err, marker="s", linewidth=2, label=prec, color=c)

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Matrix Condition Number $\kappa(A)$")
    ax1.set_ylabel("Signal-to-Noise Ratio (SNR dB)")
    ax1.set_title("Numerical Fidelity across Precision Formats")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Matrix Condition Number $\kappa(A)$")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Relative Error vs Matrix Conditioning")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "precision_error_tradeoff.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Generated: {path}")


def plot_executive_dashboard(summary_rows, precision_csv="results/benchmarks/precision_error_report.csv"):
    """Composite 4-panel executive dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("MathGPU Kernel Lab: Executive GPU Performance & Accuracy Dashboard\nNVIDIA RTX 500 Ada Generation (Ada Lovelace, sm_89)", fontsize=16)

    # Panel 1: GEMM Throughput Progression
    ax1 = axes[0, 0]
    gemm_data = [r for r in summary_rows if r["domain"] == "GEMM"]
    if gemm_data:
        k_names = [r["kernel"].replace("CUDA", "").replace("PyTorch ", "").strip() for r in gemm_data]
        tflops = [float(r["throughput_metric"].split()[0]) for r in gemm_data]
        colors = ["#1976D2", "#B0BEC5", "#78909C", "#455A64", "#7B1FA2", "#00C853"]
        bars = ax1.bar(k_names, tflops, color=colors, edgecolor="black", alpha=0.9)
        ax1.set_title("(A) GEMM Optimization Frontier (N=1024)")
        ax1.set_ylabel("Throughput (TFLOPS)")
        ax1.tick_params(axis="x", rotation=20)
        for b in bars:
            y = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2, y + 0.3, f"{y:.1f}T", ha="center", fontsize=8.5, fontweight="bold")

    # Panel 2: Attention Throughput & Latency
    ax2 = axes[0, 1]
    att_data = [r for r in summary_rows if r["domain"] == "Attention"]
    if att_data:
        att_names = [r["kernel"].replace("Attention", "").replace("CUDA", "").strip() for r in att_data]
        att_tflops = [float(r["throughput_metric"].split()[0]) for r in att_data]
        bars2 = ax2.bar(att_names, att_tflops, color=["#37474F", "#78909C", "#90A4AE", "#FF5252"], edgecolor="black", alpha=0.9)
        ax2.set_title("(B) Attention Scaled Throughput (B=8, S=1024, D=64)")
        ax2.set_ylabel("Throughput (TFLOPS)")
        ax2.tick_params(axis="x", rotation=15)
        for b in bars2:
            y = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2, y + 0.15, f"{y:.2f}T", ha="center", fontsize=8.5, fontweight="bold")

    # Panel 3: Precision SNR vs Condition Number
    ax3 = axes[1, 0]
    if os.path.exists(precision_csv):
        with open(precision_csv, "r") as f:
            p_rows = [r for r in csv.DictReader(f) if r["size"] == "512"]
        precisions = sorted(list(set(r["precision"] for r in p_rows)))
        color_map = {"FP32 (IEEE)": "#2E7D32", "TF32": "#1565C0", "FP16": "#E65100", "BF16": "#C2185B"}
        for prec in precisions:
            sub = [r for r in p_rows if r["precision"] == prec]
            sub.sort(key=lambda x: float(x["condition_number"]))
            ax3.plot([float(r["condition_number"]) for r in sub], [float(r["snr_db"]) for r in sub], marker="o", label=prec, color=color_map.get(prec, "black"), linewidth=2)
        ax3.set_xscale("log")
        ax3.set_title(r"(C) Numerical Fidelity vs Matrix Conditioning $\kappa(A)$")
        ax3.set_xlabel(r"Condition Number $\kappa(A)$")
        ax3.set_ylabel("SNR (dB)")
        ax3.legend()

    # Panel 4: Best-in-Class Speedups vs PyTorch Baselines
    ax4 = axes[1, 1]
    domains = ["GEMM", "Gaussian RBF", "KAN Layer", "Softmax", "Attention", "Tensor Contraction"]
    best_speedups = []
    best_names = []
    for d in domains:
        dom_rows = [r for r in summary_rows if r["domain"] == d]
        if dom_rows:
            best_r = max(dom_rows, key=lambda x: float(x["speedup_vs_baseline"]))
            best_speedups.append(float(best_r["speedup_vs_baseline"]))
            best_names.append(f"{d}\n({best_r['kernel']})")

    bars4 = ax4.bar(range(len(best_speedups)), best_speedups, color="#00C853", edgecolor="black", alpha=0.85)
    ax4.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="PyTorch Baseline (1.0x)")
    ax4.set_xticks(range(len(best_speedups)))
    ax4.set_xticklabels(best_names, rotation=25, ha="right", fontsize=8.5)
    ax4.set_title("(D) Peak Speedup Achieved vs Native PyTorch")
    ax4.set_ylabel("Speedup Factor (x)")
    ax4.legend()
    for b in bars4:
        y = b.get_height()
        ax4.text(b.get_x() + b.get_width()/2, y + 0.15, f"{y:.2f}x", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(OUT_DIR, "executive_dashboard.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Generated: {path}")


def main():
    summary_csv = "results/benchmarks/end_to_end_benchmark_summary.csv"
    precision_csv = "results/benchmarks/precision_error_report.csv"

    if not os.path.exists(summary_csv):
        print(f"Error: {summary_csv} not found! Run benchmarks/run_all.py first.")
        return

    with open(summary_csv, "r") as f:
        summary_rows = list(csv.DictReader(f))

    plot_gemm_throughput(summary_rows)
    plot_speedups(summary_rows)
    plot_precision_tradeoffs(precision_csv)
    plot_executive_dashboard(summary_rows, precision_csv)


if __name__ == "__main__":
    main()
