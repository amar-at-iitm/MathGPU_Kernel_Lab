"""
Test dynamic CUDA JIT compilation using NVRTC and driver launch.
"""

import pytest
import torch
from cuda.include.jit_cuda import get_cuda_jit_engine


def test_jit_cuda_vector_add():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    engine = get_cuda_jit_engine()
    cuda_source = """
    extern "C" __global__ void vec_add(float* c, const float* a, const float* b, int n) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    """
    func = engine.compile_and_load(cuda_source, "vec_add")

    n = 2048
    a = torch.ones(n, dtype=torch.float32, device="cuda") * 3.0
    b = torch.ones(n, dtype=torch.float32, device="cuda") * 4.0
    c = torch.empty(n, dtype=torch.float32, device="cuda")

    block = (256, 1, 1)
    grid = ((n + 255) // 256, 1, 1)

    engine.launch(func, grid, block, [c, a, b, n])
    torch.cuda.synchronize()

    expected = torch.ones(n, dtype=torch.float32, device="cuda") * 7.0
    torch.testing.assert_close(c, expected)
