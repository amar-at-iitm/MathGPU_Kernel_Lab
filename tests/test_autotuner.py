import pytest
import torch
from optimization.autotuning.search_space import (
    KernelSearchSpace,
    create_triton_gemm_search_space,
    create_cuda_tiled_search_space,
)
from optimization.autotuning.tuner import (
    KernelAutotuner,
    tune_triton_gemm,
)


class TestKernelSearchSpace:
    def test_search_space_grid(self):
        space = KernelSearchSpace()
        space.add_param("A", [1, 2])
        space.add_param("B", [10, 20, 30])
        grid = space.get_grid()
        assert len(grid) == 6

    def test_search_space_constraints(self):
        space = KernelSearchSpace(constraint_fn=lambda cfg: cfg["A"] != cfg["B"])
        space.add_param("A", [1, 2])
        space.add_param("B", [1, 2])
        grid = space.get_grid()
        assert len(grid) == 2
        assert {"A": 1, "B": 2} in grid
        assert {"A": 2, "B": 1} in grid

    def test_triton_gemm_search_space(self):
        space = create_triton_gemm_search_space()
        samples = space.sample_random(num_samples=5)
        assert len(samples) == 5
        for s in samples:
            assert space.is_valid(s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestKernelAutotuner:
    def test_random_tuning_gemm(self):
        summary = tune_triton_gemm(M=256, N=256, K=256, strategy="random", n_trials=3)
        assert summary.best_time_ms > 0
        assert summary.speedup >= 1.0
        assert len(summary.trials) == 3
        assert "BLOCK_SIZE_M" in summary.best_config

    def test_bayesian_tuning_gemm(self):
        summary = tune_triton_gemm(M=256, N=256, K=256, strategy="bayesian", n_trials=3)
        assert summary.best_time_ms > 0
        assert len(summary.trials) == 3
        assert "BLOCK_SIZE_M" in summary.best_config
