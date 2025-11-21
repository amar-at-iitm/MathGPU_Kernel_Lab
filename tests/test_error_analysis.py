import pytest
import torch
from numerical_analysis.error_analysis import (
    generate_conditioned_matrix,
    evaluate_precision_modes,
    run_precision_benchmark,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestErrorAnalysis:
    def test_condition_number_generation(self):
        n = 64
        cond = 100.0
        A = generate_conditioned_matrix(n, cond, device="cpu")
        _, S, _ = torch.linalg.svd(A)
        actual_cond = (S[0] / S[-1]).item()
        assert abs(actual_cond - cond) / cond < 0.05

    def test_precision_error_hierarchy(self):
        results = evaluate_precision_modes(n=128, condition_number=10.0)
        res_by_prec = {r["precision"]: r for r in results}

        assert "FP32 (IEEE)" in res_by_prec
        assert "TF32" in res_by_prec
        assert "FP16" in res_by_prec
        assert "BF16" in res_by_prec

        # FP32 strict should have lower relative error and higher SNR than FP16/BF16
        assert res_by_prec["FP32 (IEEE)"]["relative_error"] < res_by_prec["FP16"]["relative_error"]
        assert res_by_prec["FP32 (IEEE)"]["snr_db"] > res_by_prec["FP16"]["snr_db"]

    def test_run_precision_benchmark(self, tmp_path):
        out_csv = tmp_path / "test_precision.csv"
        rows = run_precision_benchmark(sizes=[64], condition_numbers=[1.0, 10.0], output_csv=str(out_csv))
        assert len(rows) == 8
        assert out_csv.exists()
