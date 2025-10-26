import pytest
import torch
from cuda.optimized.tensor_contraction_runner import cuda_tensor_contraction_4d


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTensorContraction4D:
    def test_basic_contraction(self):
        I, J, K_dim, L = 4, 8, 4, 8
        A_dim, B_dim = 6, 6

        torch.manual_seed(42)
        A = torch.randn((I, J, A_dim, B_dim), device="cuda", dtype=torch.float32)
        B = torch.randn((A_dim, B_dim, K_dim, L), device="cuda", dtype=torch.float32)

        C_ref = torch.einsum("ijab,abkl->ijkl", A, B)
        C_cuda = cuda_tensor_contraction_4d(A, B)

        assert C_cuda.shape == (I, J, K_dim, L)
        assert torch.allclose(C_cuda, C_ref, atol=1e-4, rtol=1e-4)

    def test_non_power_of_two_shapes(self):
        I, J, K_dim, L = 3, 7, 5, 9
        A_dim, B_dim = 7, 11

        torch.manual_seed(123)
        A = torch.randn((I, J, A_dim, B_dim), device="cuda", dtype=torch.float32)
        B = torch.randn((A_dim, B_dim, K_dim, L), device="cuda", dtype=torch.float32)

        C_ref = torch.einsum("ijab,abkl->ijkl", A, B)
        C_cuda = cuda_tensor_contraction_4d(A, B)

        assert torch.allclose(C_cuda, C_ref, atol=1e-4, rtol=1e-4)

    def test_larger_tensor(self):
        I, J, K_dim, L = 8, 16, 8, 16
        A_dim, B_dim = 16, 16

        torch.manual_seed(999)
        A = torch.randn((I, J, A_dim, B_dim), device="cuda", dtype=torch.float32)
        B = torch.randn((A_dim, B_dim, K_dim, L), device="cuda", dtype=torch.float32)

        C_ref = torch.einsum("ijab,abkl->ijkl", A, B)
        C_cuda = cuda_tensor_contraction_4d(A, B)

        assert torch.allclose(C_cuda, C_ref, atol=1e-4, rtol=1e-4)
