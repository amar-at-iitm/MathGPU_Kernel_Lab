"""
Tests for Nsight profiling script command builders and configuration loaders.
"""

from profiling.nsight_systems.profile_sys import build_nsys_command
from profiling.nsight_compute.profile_ncu import build_ncu_command, load_metrics_list


def test_nsys_command_builder():
    cmd = build_nsys_command(["python", "test.py"], output_report="profiles/test_trace")
    assert any("--output=profiles/test_trace" in arg for arg in cmd)
    assert any("--trace=cuda,nvtx,osrt" in arg for arg in cmd)
    assert "python" in cmd
    assert "test.py" in cmd


def test_ncu_metrics_loading():
    metrics = load_metrics_list()
    assert len(metrics) > 0
    assert any("sm__warps_active" in m for m in metrics)


def test_ncu_command_builder():
    cmd = build_ncu_command(["python", "test.py"], output_report="profiles/test_ncu", kernel_regex="gemm.*")
    assert any("--export=profiles/test_ncu" in arg for arg in cmd)
    assert "-k" in cmd
    assert "gemm.*" in cmd
