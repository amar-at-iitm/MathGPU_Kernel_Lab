"""
Tests for Roofline Model and arithmetic intensity calculations.
"""

import os
import pytest
from profiling.roofline.roofline import RooflineModel, get_device_spec


def test_device_spec_retrieval():
    """Verify device specification dictionary contains required keys."""
    spec = get_device_spec(0)
    assert "peak_fp32_tflops" in spec
    assert "peak_memory_bandwidth_gbps" in spec
    assert spec["peak_fp32_tflops"] > 0
    assert spec["peak_memory_bandwidth_gbps"] > 0


def test_roofline_classification():
    """Verify memory-bound vs compute-bound classification."""
    spec = {
        "device_name": "Test GPU",
        "peak_fp32_tflops": 10.0,
        "peak_fp16_tensor_tflops": 40.0,
        "peak_memory_bandwidth_gbps": 100.0,
    }
    model = RooflineModel(spec)
    # Ridge point = 10 * 1000 / 100 = 100 FLOPs/Byte
    assert abs(model.ridge_point_fp32 - 100.0) < 1e-4

    # AI = 10 FLOPs/Byte -> memory bound
    assert model.classify_operation(10.0, "fp32") == "memory-bound"
    # AI = 200 FLOPs/Byte -> compute bound
    assert model.classify_operation(200.0, "fp32") == "compute-bound"

    # Attainable TFLOPS test:
    # At AI = 10: 10 * 100 / 1000 = 1.0 TFLOPS
    assert abs(model.attainable_tflops(10.0, "fp32") - 1.0) < 1e-4
    # At AI = 500: capped at peak = 10.0 TFLOPS
    assert abs(model.attainable_tflops(500.0, "fp32") - 10.0) < 1e-4


def test_roofline_plot_generation(tmp_path):
    """Verify roofline plot is generated without error."""
    model = RooflineModel()
    out_file = str(tmp_path / "test_roofline.png")
    model.plot_roofline(
        operating_points=[{"name": "TestKernel", "ai": 20.0, "tflops": 2.0}],
        output_path=out_file
    )
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000
