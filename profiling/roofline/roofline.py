"""
Roofline Analysis Engine for Modern NVIDIA GPU Architectures.
Computes theoretical bounds, ridge points, and visualizes kernel arithmetic intensity.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_gpu_specs() -> Dict[str, Any]:
    json_path = os.path.join(os.path.dirname(__file__), "gpu_specs.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


def get_device_spec(device_idx: int = 0) -> Dict[str, Any]:
    """Retrieve hardware specs for the specified device."""
    specs = load_gpu_specs()
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(device_idx)
        # Check for exact or substring match
        for key, val in specs.items():
            if key in dev_name or dev_name in key:
                val_copy = dict(val)
                val_copy["device_name"] = dev_name
                return val_copy
        # Generic fallback based on device properties
        prop = torch.cuda.get_device_properties(device_idx)
        return {
            "device_name": dev_name,
            "architecture": f"CC {prop.major}.{prop.minor}",
            "compute_capability": f"{prop.major}.{prop.minor}",
            "peak_fp32_tflops": 10.0,
            "peak_fp16_tensor_tflops": 40.0,
            "peak_memory_bandwidth_gbps": 120.0,
            "vram_gb": round(prop.total_memory / (1024**3), 1),
        }
    return {
        "device_name": "CPU/Generic",
        "architecture": "Generic",
        "compute_capability": "0.0",
        "peak_fp32_tflops": 1.0,
        "peak_fp16_tensor_tflops": 2.0,
        "peak_memory_bandwidth_gbps": 50.0,
        "vram_gb": 16.0,
    }


class RooflineModel:
    """Roofline Model for compute-bound vs memory-bound classification."""

    def __init__(self, spec: Optional[Dict[str, Any]] = None):
        self.spec = spec or get_device_spec(0)
        self.peak_fp32_tflops: float = self.spec.get("peak_fp32_tflops", 10.0)
        self.peak_fp16_tflops: float = self.spec.get("peak_fp16_tensor_tflops", 40.0)
        self.bandwidth_gbps: float = self.spec.get("peak_memory_bandwidth_gbps", 112.0)
        # Ridge point in FLOPs/Byte = (TFLOPS * 10^12) / (Bandwidth * 10^9) = (TFLOPS * 1000) / Bandwidth
        self.ridge_point_fp32: float = (self.peak_fp32_tflops * 1000.0) / self.bandwidth_gbps
        self.ridge_point_fp16: float = (self.peak_fp16_tflops * 1000.0) / self.bandwidth_gbps

    def attainable_tflops(self, arithmetic_intensity: float, precision: str = "fp32") -> float:
        """
        Attainable TFLOPS = min(Peak TFLOPS, AI * Peak Bandwidth / 1000)
        """
        peak = self.peak_fp16_tflops if precision == "fp16" else self.peak_fp32_tflops
        memory_bound_tflops = (arithmetic_intensity * self.bandwidth_gbps) / 1000.0
        return min(peak, memory_bound_tflops)

    def classify_operation(self, arithmetic_intensity: float, precision: str = "fp32") -> str:
        """Classify operation as 'memory-bound' or 'compute-bound'."""
        ridge = self.ridge_point_fp16 if precision == "fp16" else self.ridge_point_fp32
        return "memory-bound" if arithmetic_intensity < ridge else "compute-bound"

    def plot_roofline(self,
                      operating_points: Optional[List[Dict[str, Any]]] = None,
                      output_path: str = "results/plots/roofline.png") -> None:
        """
        Generate and save a visual Roofline plot.
        operating_points: List of dicts with 'name', 'ai', 'tflops'.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        ai_range = np.logspace(-1, 3, 500)  # 0.1 to 1000 FLOPs/Byte

        # Compute curves
        roof_fp32 = [self.attainable_tflops(ai, "fp32") for ai in ai_range]
        roof_fp16 = [self.attainable_tflops(ai, "fp16") for ai in ai_range]

        plt.figure(figsize=(10, 6))
        plt.loglog(ai_range, roof_fp16, label=f"Peak FP16 Tensor ({self.peak_fp16_tflops:.1f} TFLOPS)",
                   color="#e65100", linewidth=2.2, linestyle="--")
        plt.loglog(ai_range, roof_fp32, label=f"Peak FP32 ({self.peak_fp32_tflops:.1f} TFLOPS)",
                   color="#1565c0", linewidth=2.2)

        # Plot Ridge points
        plt.axvline(self.ridge_point_fp32, color="#1565c0", linestyle=":", alpha=0.6,
                    label=f"FP32 Ridge Point ({self.ridge_point_fp32:.1f} FLOPs/B)")

        # Plot kernel operating points if supplied
        if operating_points:
            colors = ["#2e7d32", "#c2185b", "#6a1b9a", "#00838f", "#d84315"]
            for i, pt in enumerate(operating_points):
                c = colors[i % len(colors)]
                plt.scatter(pt["ai"], pt["tflops"], color=c, s=120, zorder=5, edgecolors="black")
                plt.annotate(pt["name"], (pt["ai"] * 1.08, pt["tflops"] * 0.95),
                             fontsize=9, weight="bold", color=c)

        plt.title(f"Roofline Model — {self.spec.get('device_name', 'GPU')} "
                  f"(BW: {self.bandwidth_gbps:.0f} GB/s)", fontsize=13, weight="bold")
        plt.xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=11)
        plt.ylabel("Performance (TFLOPS)", fontsize=11)
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[Roofline] Model plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate GPU Roofline Analysis")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--out", type=str, default="results/plots/roofline_baseline.png", help="Output path")
    args = parser.parse_args()

    spec = get_device_spec(args.device)
    print(f"\n[Hardware Specs] {spec.get('device_name')}")
    print(f"  - Peak FP32: {spec.get('peak_fp32_tflops')} TFLOPS")
    print(f"  - Peak FP16 Tensor: {spec.get('peak_fp16_tensor_tflops')} TFLOPS")
    print(f"  - Peak Bandwidth: {spec.get('peak_memory_bandwidth_gbps')} GB/s")

    model = RooflineModel(spec)
    print(f"  - FP32 Ridge Point: {model.ridge_point_fp32:.2f} FLOPs/Byte")
    print(f"  - FP16 Ridge Point: {model.ridge_point_fp16:.2f} FLOPs/Byte\n")

    sample_points = [
        {"name": "cuBLAS GEMM FP32 (2048x2048)", "ai": 85.3, "tflops": 6.8},
        {"name": "Naive GEMM (2048x2048)", "ai": 0.5, "tflops": 0.05},
        {"name": "Tiled GEMM (2048x2048)", "ai": 16.0, "tflops": 1.7},
    ]
    model.plot_roofline(sample_points, args.out)


if __name__ == "__main__":
    main()
