"""
High-Precision CUDA Event Timing & Benchmarking Module.
Provides microsecond-accurate GPU execution timers without CPU synchronization overhead.
"""

from typing import Callable, Any, Dict, List, Tuple
import time
import numpy as np
import torch


class CudaTimer:
    """Context manager for measuring GPU elapsed time using CUDA Events."""

    def __init__(self, stream: torch.cuda.Stream = None):
        self.stream = stream or torch.cuda.current_stream()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stop_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_event.record(self.stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.record(self.stream)
        self.stop_event.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.stop_event)


def benchmark_function(fn: Callable[..., Any],
                       *args: Any,
                       warmup: int = 15,
                       rep: int = 50,
                       **kwargs: Any) -> Dict[str, float]:
    """
    Benchmark a callable using GPU CUDA Events.

    Args:
        fn: Function or callable to benchmark.
        *args: Positional arguments for fn.
        warmup: Number of initial iterations to disregard (warm caches, clock boost).
        rep: Number of active measurement iterations.
        **kwargs: Keyword arguments for fn.

    Returns:
        Dictionary with 'mean_ms', 'median_ms', 'std_ms', 'min_ms', 'max_ms', 'p95_ms', 'p99_ms'.
    """
    use_cuda = torch.cuda.is_available()

    # Warmup phase
    for _ in range(warmup):
        fn(*args, **kwargs)
    if use_cuda:
        torch.cuda.synchronize()

    timings_ms: List[float] = []

    if use_cuda:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

        for i in range(rep):
            start_events[i].record()
            fn(*args, **kwargs)
            stop_events[i].record()

        torch.cuda.synchronize()
        timings_ms = [start_events[i].elapsed_time(stop_events[i]) for i in range(rep)]
    else:
        # Fallback to CPU high-resolution timer
        for _ in range(rep):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    arr = np.array(timings_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }
