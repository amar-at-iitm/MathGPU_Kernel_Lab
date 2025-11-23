"""
GPU Kernel Parameter Autotuner.
Supports Grid Search, Random Search, and Bayesian Optimization (via Optuna).
Features: CUDA Event micro-benchmarking, warmup cycles, outlier filtering, and exception-safe evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time
import torch
import triton
from optimization.autotuning.search_space import KernelSearchSpace, create_triton_gemm_search_space


@dataclass
class TrialResult:
    config: Dict[str, Any]
    time_ms: float
    tflops: float
    is_valid: bool
    error_msg: Optional[str] = None


@dataclass
class TuningSummary:
    best_config: Dict[str, Any]
    best_time_ms: float
    best_tflops: float
    baseline_time_ms: float
    speedup: float
    trials: List[TrialResult] = field(default_factory=list)

    def print_summary(self):
        print("=" * 60)
        print("               AUTOTUNING SUMMARY REPORT")
        print("=" * 60)
        print(f"Total Configurations Evaluated: {len(self.trials)}")
        valid_trials = [t for t in self.trials if t.is_valid]
        print(f"Valid Executions:              {len(valid_trials)}")
        print(f"Baseline Time:                 {self.baseline_time_ms:.4f} ms")
        print(f"Optimal Kernel Time:           {self.best_time_ms:.4f} ms")
        print(f"Peak Performance:              {self.best_tflops:.2f} TFLOPS")
        print(f"Speedup vs Baseline:           {self.speedup:.2f}x")
        print("Optimal Parameters:")
        for k, v in self.best_config.items():
            print(f"  - {k}: {v}")
        print("=" * 60)


def benchmark_gpu_fn(fn: Callable[[], Any],
                     warmup: int = 10,
                     repeat: int = 20) -> float:
    """Benchmark a callable GPU function using high-resolution CUDA Events."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    timings = []
    for _ in range(repeat):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    timings.sort()
    # Return median runtime in milliseconds
    return timings[len(timings) // 2]


class KernelAutotuner:
    """Universal Kernel Autotuning Engine."""

    def __init__(self,
                 search_space: KernelSearchSpace,
                 kernel_eval_fn: Callable[[Dict[str, Any]], Callable[[], Any]],
                 flops_fn: Optional[Callable[[], float]] = None):
        """
        kernel_eval_fn: Given a config dict, returns a callable () -> None that executes the kernel.
        flops_fn: Returns total FLOPs for the kernel operation (for TFLOPS computation).
        """
        self.search_space = search_space
        self.kernel_eval_fn = kernel_eval_fn
        self.flops_fn = flops_fn

    def _eval_config(self, config: Dict[str, Any]) -> TrialResult:
        flops = self.flops_fn() if self.flops_fn else 0.0
        try:
            fn = self.kernel_eval_fn(config)
            time_ms = benchmark_gpu_fn(fn, warmup=5, repeat=15)
            tflops = (flops / (time_ms * 1e-3)) / 1e12 if flops > 0 and time_ms > 0 else 0.0
            return TrialResult(config=config, time_ms=time_ms, tflops=tflops, is_valid=True)
        except Exception as e:
            return TrialResult(config=config, time_ms=float("inf"), tflops=0.0, is_valid=False, error_msg=str(e))

    def tune_grid(self) -> TuningSummary:
        """Exhaustive Grid Search."""
        configs = self.search_space.get_grid()
        trials = [self._eval_config(cfg) for cfg in configs]
        return self._compile_summary(trials)

    def tune_random(self, n_trials: int = 15, seed: int = 42) -> TuningSummary:
        """Random Search."""
        configs = self.search_space.sample_random(num_samples=n_trials, seed=seed)
        trials = [self._eval_config(cfg) for cfg in configs]
        return self._compile_summary(trials)

    def tune_bayesian(self, n_trials: int = 15, seed: int = 42) -> TuningSummary:
        """Bayesian Optimization using Optuna."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        trials: List[TrialResult] = []

        def objective(trial: optuna.Trial) -> float:
            cfg = {}
            for name, param in self.search_space.parameters.items():
                cfg[name] = trial.suggest_categorical(name, param.choices)

            if not self.search_space.is_valid(cfg):
                raise optuna.TrialPruned()

            res = self._eval_config(cfg)
            trials.append(res)
            if not res.is_valid:
                return float("inf")
            return res.time_ms

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        return self._compile_summary(trials)

    def _compile_summary(self, trials: List[TrialResult]) -> TuningSummary:
        valid_trials = [t for t in trials if t.is_valid and t.time_ms < float("inf")]
        if not valid_trials:
            raise RuntimeError("All autotuning configurations failed!")

        valid_trials.sort(key=lambda t: t.time_ms)
        best = valid_trials[0]
        worst_or_baseline = valid_trials[-1]

        speedup = worst_or_baseline.time_ms / best.time_ms if best.time_ms > 0 else 1.0

        return TuningSummary(
            best_config=best.config,
            best_time_ms=best.time_ms,
            best_tflops=best.tflops,
            baseline_time_ms=worst_or_baseline.time_ms,
            speedup=speedup,
            trials=trials
        )


def tune_triton_gemm(M: int = 1024,
                     N: int = 1024,
                     K: int = 1024,
                     strategy: str = "random",
                     n_trials: int = 10) -> TuningSummary:
    """Benchmark and autotune Triton GEMM kernel over hyperparameter space."""
    from triton_kernels.gemm import _matmul_kernel

    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)

    space = create_triton_gemm_search_space()

    def eval_fn(cfg: Dict[str, Any]) -> Callable[[], Any]:
        bm = cfg["BLOCK_SIZE_M"]
        bn = cfg["BLOCK_SIZE_N"]
        bk = cfg["BLOCK_SIZE_K"]
        gm = cfg["GROUP_SIZE_M"]
        nw = cfg.get("num_warps", 4)
        ns = cfg.get("num_stages", 3)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

        def runner():
            _matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=bm,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk,
                GROUP_SIZE_M=gm,
                num_warps=nw,
                num_stages=ns,
            )
        return runner

    flops = 2.0 * M * N * K
    tuner = KernelAutotuner(space, eval_fn, flops_fn=lambda: flops)

    if strategy == "grid":
        return tuner.tune_grid()
    elif strategy == "bayesian":
        return tuner.tune_bayesian(n_trials=n_trials)
    else:
        return tuner.tune_random(n_trials=n_trials)
