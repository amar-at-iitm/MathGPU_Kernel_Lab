"""
Executable CLI script to autotune Triton GEMM parameters on the active GPU.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.autotuning.tuner import tune_triton_gemm


def main():
    parser = argparse.ArgumentParser(description="Autotune Triton GEMM Kernel")
    parser.add_argument("--size", type=int, default=1024, help="Matrix dimension M=N=K")
    parser.add_argument("--strategy", choices=["random", "bayesian", "grid"], default="bayesian", help="Search strategy")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    args = parser.parse_args()

    print(f"Running Autotuner for Triton GEMM (Size: {args.size}x{args.size}, Strategy: {args.strategy}, Trials: {args.trials})...")
    summary = tune_triton_gemm(M=args.size, N=args.size, K=args.size, strategy=args.strategy, n_trials=args.trials)
    summary.print_summary()


if __name__ == "__main__":
    main()
