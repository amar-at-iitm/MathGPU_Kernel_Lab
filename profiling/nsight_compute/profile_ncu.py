"""
Nsight Compute (ncu) Microarchitecture Metric Profiling Automation.
"""

import os
import shutil
import subprocess
import argparse
from typing import List, Optional


def load_metrics_list(cfg_path: Optional[str] = None) -> List[str]:
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "ncu_metrics.cfg")
    metrics = []
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    metrics.append(line)
    return metrics


def build_ncu_command(target_cmd: List[str],
                      output_report: str = "results/profiles/ncu_report",
                      metrics: Optional[List[str]] = None,
                      kernel_regex: Optional[str] = None) -> List[str]:
    """Construct command line arguments for NVIDIA Nsight Compute (ncu)."""
    ncu_bin = shutil.which("ncu") or "ncu"
    cmd = [ncu_bin, f"--export={output_report}", "--force-overwrite"]

    if kernel_regex:
        cmd.extend(["-k", kernel_regex])

    if metrics:
        cmd.extend(["--metrics", ",".join(metrics)])
    else:
        cmd.extend(["--set", "detailed"])

    cmd.extend(target_cmd)
    return cmd


def run_ncu_profile(target_cmd: List[str],
                    output_report: str = "results/profiles/ncu_report",
                    kernel_regex: Optional[str] = None) -> int:
    """Execute ncu profile or provide clear guidance if ncu CLI is not found."""
    if not shutil.which("ncu"):
        print("\n[Notice] NVIDIA Nsight Compute CLI ('ncu') not found in PATH.")
        print("To profile microarchitecture metrics, install Nsight Compute or run:")
        print(f"  ncu --set detailed -o {output_report} {' '.join(target_cmd)}\n")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    metrics = load_metrics_list()
    cmd = build_ncu_command(target_cmd, output_report, metrics=metrics, kernel_regex=kernel_regex)
    print(f"[Executing] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Automate NVIDIA Nsight Compute Kernel Profiling")
    parser.add_argument("--output", type=str, default="results/profiles/ncu_report", help="Output report path")
    parser.add_argument("-k", "--kernel-regex", type=str, default=None, help="Filter kernels by regex")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Target command to run")
    args = parser.parse_args()

    cmd = args.command if args.command else ["python", "benchmarks/gemm/benchmark_gemm_pytorch.py"]
    exit(run_ncu_profile(cmd, args.output, args.kernel_regex))


if __name__ == "__main__":
    main()
