import pytest
import torch
from triton_kernels.tensor_contraction import triton_tensor_contraction_4d
from cuda.optimized.tensor_contraction_runner import cuda_tensor_contraction_4d


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTensorContractionTriton:
    def test_triton_vs_einsum(self):
        I, J, K_dim, L = 4, 8, 4, 8
        A_dim, B_dim = 8, 8

        torch.manual_seed(42)
        scale = 1.0 / ((A_dim * B_dim) ** 0.5)
        A = torch.randn((I, J, A_dim, B_dim), device="cuda", dtype=torch.float32) * scale
        B = torch.randn((A_dim, B_dim, K_dim, L), device="cuda", dtype=torch.float32) * scale

        C_ref = torch.einsum("ijab,abkl->ijkl", A, B)
        C_triton = triton_tensor_contraction_4d(A, B)

        assert C_triton.shape == (I, J, K_dim, L)
        torch.testing.assert_close(C_triton, C_ref, rtol=1e-2, atol=1e-2)

    def test_triton_vs_cuda_kernel(self):
        I, J, K_dim, L = 8, 16, 8, 16
        A_dim, B_dim = 16, 16

        torch.manual_seed(101)
        scale = 1.0 / ((A_dim * B_dim) ** 0.5)
        A = torch.randn((I, J, A_dim, B_dim), device="cuda", dtype=torch.float32) * scale
        B = torch.randn((A_dim, B_dim, K_dim, L), device="cuda", dtype=torch.float32) * scale

        C_cuda = cuda_tensor_contraction_4d(A, B)
        C_triton = triton_tensor_contraction_4d(A, B)

        torch.testing.assert_close(C_triton, C_cuda, rtol=1e-2, atol=1e-2)

    def test_arbitrary_tensor_dimensions(self):
        I, J, K_dim, L = 5, 7, 6, 9
        A_dim, B_dim = 7, 11

        torch.manual_seed(202)
        scale = 1.0 / ((A_dim * B_dim) ** 0.5)
        A = torch.randn((I, J, A_dim, B_dim), device="cuda", dtype=torch.float32) * scale
        B = torch.randn((A_dim, B_dim, K_dim, L), device="cuda", dtype=torch.float32) * scale

        C_ref = torch.einsum("ijab,abkl->ijkl", A, B)
        C_triton = triton_tensor_contraction_4d(A, B)

        torch.testing.assert_close(C_triton, C_ref, rtol=1e-2, atol=1e-2)
