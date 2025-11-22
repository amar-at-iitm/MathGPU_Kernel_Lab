"""
Kernel Search Space Definition for Automated GPU Kernel Autotuning.
Defines parameter ranges, discrete choices, and structural constraints for CUDA and Triton kernels.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import itertools


@dataclass
class Parameter:
    name: str
    choices: List[Any]

    def sample_random(self, rng=None):
        import random
        r = rng or random
        return r.choice(self.choices)


@dataclass
class KernelSearchSpace:
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    constraint_fn: Optional[Callable[[Dict[str, Any]], bool]] = None

    def add_param(self, name: str, choices: List[Any]) -> "KernelSearchSpace":
        self.parameters[name] = Parameter(name=name, choices=choices)
        return self

    def is_valid(self, config: Dict[str, Any]) -> bool:
        if self.constraint_fn is not None:
            return self.constraint_fn(config)
        return True

    def get_grid(self) -> List[Dict[str, Any]]:
        """Generate all valid configurations in the cartesian product."""
        keys = list(self.parameters.keys())
        all_combos = itertools.product(*(self.parameters[k].choices for k in keys))
        valid_configs = []
        for combo in all_combos:
            cfg = dict(zip(keys, combo))
            if self.is_valid(cfg):
                valid_configs.append(cfg)
        return valid_configs

    def sample_random(self, num_samples: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
        """Sample valid configurations uniformly at random."""
        import random
        rng = random.Random(seed)
        grid = self.get_grid()
        if not grid:
            return []
        rng.shuffle(grid)
        return grid[:min(num_samples, len(grid))]


def create_triton_gemm_search_space() -> KernelSearchSpace:
    """Standard search space for Triton Matrix Multiplication kernels."""
    def gemm_constraint(cfg: Dict[str, Any]) -> bool:
        bm = cfg["BLOCK_SIZE_M"]
        bn = cfg["BLOCK_SIZE_N"]
        nw = cfg.get("num_warps", 4)
        # Ensure at least 1 warp per 16 elements
        threads = nw * 32
        if (bm * bn) < threads:
            return False
        return True

    space = KernelSearchSpace(constraint_fn=gemm_constraint)
    space.add_param("BLOCK_SIZE_M", [32, 64, 128])
    space.add_param("BLOCK_SIZE_N", [32, 64, 128])
    space.add_param("BLOCK_SIZE_K", [16, 32, 64])
    space.add_param("GROUP_SIZE_M", [4, 8])
    space.add_param("num_warps", [2, 4, 8])
    space.add_param("num_stages", [2, 3, 4])
    return space


def create_cuda_tiled_search_space() -> KernelSearchSpace:
    """Standard search space for Shared-Memory Tiled CUDA kernels."""
    space = KernelSearchSpace()
    space.add_param("TILE_DIM", [8, 16, 32])
    space.add_param("REG_M", [1, 2, 4])
    space.add_param("REG_N", [1, 2, 4])
    return space
