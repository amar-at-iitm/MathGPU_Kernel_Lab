"""
Nsight Systems Timeline Profiling Automation.
"""

import os
import shutil
import subprocess
import argparse
from typing import List, Optional


def build_nsys_command(target_cmd: List[str],
                       output_report: str = "results/profiles/nsys_trace",
                       trace: str = "cuda,nvtx,osrt",
                       force_overwrite: bool = True) -> List[str]:
    """Construct command line arguments for NVIDIA Nsight Systems (nsys)."""
    nsys_bin = shutil.which("nsys") or "nsys"
    cmd = [
        nsys_bin, "profile",
        f"--output={output_report}",
        f"--trace={trace}",
        "--cuda-memory-usage=true",
        "--stats=true",
    ]
    if force_overwrite:
        cmd.append("--force-overwrite=true")
    cmd.extend(target_cmd)
    return cmd


def run_nsys_profile(target_cmd: List[str],
                     output_report: str = "results/profiles/nsys_trace") -> int:
    """Execute nsys profile or provide clear instructions if nsys is not installed."""
    if not shutil.which("nsys"):
        print("\n[Notice] NVIDIA Nsight Systems CLI ('nsys') not found in PATH.")
        print("To profile, install NVIDIA Nsight Systems or run:")
        print(f"  nsys profile --trace=cuda,nvtx -o {output_report} {' '.join(target_cmd)}\n")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    cmd = build_nsys_command(target_cmd, output_report)
    print(f"[Executing] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Automate NVIDIA Nsight Systems Profiling")
    parser.add_argument("--output", type=str, default="results/profiles/nsys_trace", help="Output report base path")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Target command to run under nsys")
    args = parser.parse_args()

    cmd = args.command if args.command else ["python", "benchmarks/gemm/benchmark_gemm_pytorch.py"]
    exit(run_nsys_profile(cmd, args.output))


if __name__ == "__main__":
    main()
