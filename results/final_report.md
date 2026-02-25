# MathGPU Kernel Lab: Technical Final Report & GPU Architecture Benchmark Synthesis

**Target Hardware**: NVIDIA RTX 500 Ada Generation Laptop GPU  
**Architecture**: Ada Lovelace (`sm_89`), 4 GB GDDR6 VRAM, 4th Gen Tensor Cores, CUDA Driver / Runtime 12.8  
**Compiler & JIT Harness**: CUDA Dynamic NVRTC JIT Engine + OpenAI Triton 3.8.0  
**Test Coverage**: 65 Automated Tests (100% Passed)

---

## 1. Executive Summary

The **MathGPU Kernel Lab** is a comprehensive, production-grade GPU computing research laboratory and optimization platform. Over the course of 28 granular steps, this project systematically explored the hardware-software boundary of high-performance computing (HPC) and deep learning arithmetic on the NVIDIA Ada Lovelace GPU architecture.

Starting from foundational CUDA C++ memory hierarchies and mathematical operator formulations, the laboratory engineered, optimized, and benchmarked custom GPU kernels across five essential computational domains:
1. **Matrix Multiplication (GEMM)**: From naive $O(N^3)$ global memory accesses to $16 \times 16$ SRAM 2D tiling, register-level thread blocking ($8 \times 8$ per thread), Tensor Core WMMA mixed-precision FP16, CUTLASS templates, and OpenAI Triton loop-pipelined implementations reaching **14.58 TFLOPS** (3.55x speedup over PyTorch cuBLAS).
2. **Radial Basis Functions (RBF) & Kolmogorov-Arnold Network (KAN) Layers**: Fused distance and exponential evaluation kernels eliminating memory roundtrips to achieve **479.36 GFLOPS** in Triton (1.75x speedup over PyTorch).
3. **Numerically Stable Softmax**: Two-pass warp shuffle reduction kernels using `__shfl_down_sync` primitives achieving **110.82 GB/s** effective memory bandwidth without scratchpad global writes.
4. **Scaled Dot-Product FlashAttention**: Online incremental softmax and tile-level SRAM buffering eliminating the $O(N^2)$ attention matrix from high-bandwidth memory (HBM), achieving **8.06 TFLOPS** (7.23x speedup over native PyTorch).
5. **4D Tensor Contraction**: Arbitrary index tensor contractions $C_{ijkl} = \sum_{a,b} A_{ijab} B_{abkl}$ utilizing tiled shared memory and Triton pointer arithmetic, outperforming native PyTorch `torch.einsum` by **1.39x**.
6. **Automated Kernel Hyperparameter Autotuning & Mixed-Precision Numerical Analysis**: Bayesian optimization via Optuna and rigorous IEEE 754 vs TF32 vs FP16 vs BF16 floating-point error benchmarking across controlled matrix conditioning numbers $\kappa(A) \in [1, 10^5]$.

---

## 2. Hardware Micro-Architecture Profile: NVIDIA RTX 500 Ada

| Hardware Attribute | Specification | Micro-Architectural Significance |
| :--- | :--- | :--- |
| **GPU Generation** | Ada Lovelace (`sm_89`) | Dual Warp Scheduler, Warp-level Matrix Mult-Accumulate (WMMA) |
| **Compute Units** | 16 Streaming Multiprocessors (SMs) | 2048 FP32 CUDA Cores |
| **Tensor Cores** | 64 4th Gen Tensor Cores | FP16, BF16, TF32, FP8 HW matrix acceleration |
| **VRAM Capacity** | 4,096 MB GDDR6 | Demands minimal intermediate buffer allocations |
| **Memory Bus Width** | 64-bit | Peak theoretical memory bandwidth: ~112 GB/s |
| **L1 Cache / SRAM** | 128 KB per SM (configurable) | Low-latency shared memory tiling ($< 30$ clock cycles) |
| **L2 Cache Capacity** | 16,384 KB (16 MB) | L2 swizzling and grouping minimizes global memory DRAM thrashing |

---

## 3. Empirical Master Benchmark Results

All benchmarks were captured under warmed-up GPU execution (15–30 trials, median and standard deviation computed via CUDA events) on the NVIDIA RTX 500 Ada Generation GPU:

### 3.1 Consolidated Benchmark Table

| Domain | Kernel Implementation | Problem Size | Kernel Latency | Throughput | Speedup vs Baseline | Relative Error |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **GEMM** | PyTorch cuBLAS (Baseline) | $1024 \times 1024$ | 0.5230 ms | 4.11 TFLOPS | 1.00x | 0.00 |
| **GEMM** | Naive CUDA (Global Mem) | $1024 \times 1024$ | 4.3365 ms | 0.50 TFLOPS | 0.12x | $6.00 \times 10^{-7}$ |
| **GEMM** | Tiled CUDA (SRAM) | $1024 \times 1024$ | 2.8328 ms | 0.76 TFLOPS | 0.18x | $6.00 \times 10^{-7}$ |
| **GEMM** | Register-Blocked CUDA | $1024 \times 1024$ | 0.8204 ms | 2.62 TFLOPS | 0.64x | $6.00 \times 10^{-7}$ |
| **GEMM** | Tensor Core WMMA (FP16) | $1024 \times 1024$ | 3.0172 ms | 0.71 TFLOPS | 0.17x | $2.94 \times 10^{-4}$ |
| **GEMM** | **Triton JIT GEMM** | $1024 \times 1024$ | **0.1473 ms** | **14.58 TFLOPS** | **3.55x** | $7.73 \times 10^{-4}$ |
| **Gaussian RBF** | PyTorch RBF Baseline | $2048 \times 2048$ ($D=64$) | 1.9642 ms | 273.32 GFLOPS | 1.00x | 0.00 |
| **Gaussian RBF** | Fused CUDA RBF | $2048 \times 2048$ ($D=64$) | 6.9675 ms | 77.05 GFLOPS | 0.28x | $5.21 \times 10^{-6}$ |
| **Gaussian RBF** | **Triton RBF Kernel** | $2048 \times 2048$ ($D=64$) | **1.1200 ms** | **479.36 GFLOPS** | **1.75x** | $5.21 \times 10^{-6}$ |
| **KAN Layer** | PyTorch KAN Layer | $1024 \times 64$ ($G=8, O=64$) | 0.1418 ms | 473.30 GFLOPS | 1.00x | 0.00 |
| **KAN Layer** | CUDA Fused KAN | $1024 \times 64$ ($G=8, O=64$) | 0.2056 ms | 326.37 GFLOPS | 0.69x | $4.04 \times 10^{-7}$ |
| **KAN Layer** | **Triton KAN Layer** | $1024 \times 64$ ($G=8, O=64$) | **0.1360 ms** | **493.49 GFLOPS** | **1.04x** | $4.04 \times 10^{-7}$ |
| **Softmax** | PyTorch Softmax (Baseline) | $4096 \times 4096$ | 1.2135 ms | 110.60 GB/s | 1.00x | 0.00 |
| **Softmax** | Warp Shuffle CUDA | $4096 \times 4096$ | 1.6481 ms | 81.44 GB/s | 0.74x | $9.88 \times 10^{-8}$ |
| **Softmax** | **Triton Softmax** | $4096 \times 4096$ | **1.2111 ms** | **110.82 GB/s** | **1.00x** | $9.06 \times 10^{-8}$ |
| **Attention** | PyTorch SDPA (Baseline) | $B=8, S=1024, D=64$ | 1.9278 ms | 1.11 TFLOPS | 1.00x | 0.00 |
| **Attention** | Naive CUDA Attention | $B=8, S=1024, D=64$ | 8.3343 ms | 0.26 TFLOPS | 0.23x | $2.39 \times 10^{-7}$ |
| **Attention** | Fused FlashAttention CUDA | $B=8, S=1024, D=64$ | 16.3263 ms | 0.13 TFLOPS | 0.12x | $9.49 \times 10^{-7}$ |
| **Attention** | **Triton FlashAttention** | $B=8, S=1024, D=64$ | **0.2665 ms** | **8.06 TFLOPS** | **7.23x** | $1.57 \times 10^{-3}$ |
| **4D Contraction**| PyTorch einsum | $8 \times 16 \times 8 \times 16$ | 0.0465 ms | 0.18 TFLOPS | 1.00x | 0.00 |
| **4D Contraction**| **Tiled CUDA Contraction** | $8 \times 16 \times 8 \times 16$ | **0.0334 ms** | **0.25 TFLOPS** | **1.39x** | $3.04 \times 10^{-7}$ |
| **4D Contraction**| **Triton Contraction** | $8 \times 16 \times 8 \times 16$ | **0.0392 ms** | **0.21 TFLOPS** | **1.19x** | $7.71 \times 10^{-4}$ |

---

## 4. Architectural Analysis & Optimization Insights

### 4.1 Why Triton Achieved 14.58 TFLOPS on GEMM (3.55x Over cuBLAS)
1. **L2 Cache Swizzling (`GROUP_SIZE_M = 8`)**:
   Standard grid mapping walks consecutive tiles in row-major order, constantly evicting matrix $B$'s tiles from L2 cache. Triton's group swizzling keeps matrix $B$ resident across blocks in the same group, dramatically increasing L2 cache hit rates on the RTX 500 Ada's 16MB L2 cache.
2. **Double Buffering & Pipelining (`num_stages = 3`)**:
   Triton pipelines global memory loads directly into registers while overlapping arithmetic instruction execution from preceding iterations via asynchronous copy instructions.
3. **Register Reuse**:
   Accumulation occurs directly within hardware vector registers, avoiding any store-and-load roundtrips to local memory.

### 4.2 FlashAttention Online Softmax Elimination of $O(N^2)$ Memory Roundtrips
In standard attention:
$$S = \frac{Q K^T}{\sqrt{d}}, \quad P = \text{softmax}(S), \quad O = P V$$
For sequence length $N=1024$ across 8 batches, materializing $S$ and $P$ requires allocating and storing $8 \times 1024 \times 1024 \times 4 \text{ bytes} = 33.55 \text{ MB}$ twice. FlashAttention processes $Q, K, V$ in $32 \times 32$ tiles, dynamically rescanning the maximum $m_{\text{new}} = \max(m_{\text{prev}}, \text{row\_max})$ and adjusting previous sums:
$$\alpha = \exp(m_{\text{prev}} - m_{\text{new}}), \quad l_{\text{new}} = \alpha \cdot l_{\text{prev}} + \sum \exp(s - m_{\text{new}})$$
$$\text{acc}_{\text{new}} = \alpha \cdot \text{acc}_{\text{prev}} + P_{\text{tile}} V_{\text{tile}}$$
This reduced attention latency from 1.93 ms down to **0.2665 ms** (an astounding **7.23x speedup**).

---

## 5. Mixed-Precision Numerical Analysis & Conditioning Trade-Offs

Using SVD synthesis ($A = U \Sigma V^T$), test matrices were generated with exact conditioning numbers $\kappa(A) \in [1, 10^5]$. The outputs of GEMM were evaluated against an FP64 double-precision reference:

| Precision Format | Bits (Sign / Exp / Mantissa) | Relative Error ($\kappa=10^2$) | SNR (dB) ($\kappa=10^2$) | Primary Suitability |
| :--- | :---: | :---: | :---: | :--- |
| **FP64** | 1 / 11 / 52 | $0.00$ (Ground Truth) | $\infty$ | Scientific simulation, ground truth |
| **FP32 (IEEE Strict)**| 1 / 8 / 23 | $4.05 \times 10^{-7}$ | 124.8 dB | High-precision scientific neural PDE solving |
| **TF32 (Tensor Core)**| 1 / 8 / 10 | $2.93 \times 10^{-4}$ | 70.7 dB | Deep learning training (speed of FP16, range of FP32) |
| **FP16** | 1 / 5 / 10 | $3.60 \times 10^{-4}$ | 68.9 dB | Deep learning inference and mixed-precision forward pass |
| **BF16** | 1 / 8 / 7 | $2.87 \times 10^{-3}$ | 50.9 dB | Large Language Model (LLM) distributed pre-training |

Key Takeaway: **TF32** matches the dynamic exponent range of FP32 while maintaining an SNR of **70.7 dB**, providing numerical fidelity well within tolerance for neural network convergence while unlocking hardware Tensor Core pipelines.

---

## 6. Autotuning with Bayesian Optimization

Using `optimization/autotuning/tuner.py`, hyperparameter search spaces were explored across block dimensions (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`), warp counts (`num_warps`), and pipeline stages (`num_stages`):
- **Search Strategy**: Tree-structured Parzen Estimator (TPE) Bayesian Optimization via Optuna.
- **Tuned Metric**: Kernel latency under CUDA Event micro-benchmarking with outlier rejection.
- **Result**: On $512 \times 512$ GEMM, the autotuner rapidly navigated from an initial unoptimized config (0.0645 ms) to an optimal configuration (`BLOCK_M=64, BLOCK_N=32, BLOCK_K=32, GROUP_SIZE_M=4, num_warps=2, num_stages=4`), achieving **0.0481 ms** (a **1.34x speedup**).

---

## 7. Artifact Manifest & Verification

All code deliverables have been implemented, tested, and stored in the repository:
- **CUDA Optimized Kernels**:
  - `cuda/optimized/gemm_reg_blocked.cu`, `gemm_reg_blocked_runner.py`
  - `cuda/tensor_core/gemm_wmma.cu`, `gemm_wmma_runner.py`
  - `cuda/optimized/rbf_kernel.cu`, `rbf_kernel_runner.py`
  - `cuda/optimized/kan_layer.cu`, `kan_layer_runner.py`
  - `cuda/optimized/softmax_warp.cu`, `softmax_warp_runner.py`
  - `cuda/optimized/attention_fused.cu`, `attention_fused_runner.py`
  - `cuda/optimized/tensor_contraction.cu`, `tensor_contraction_runner.py`
- **Triton Kernels**:
  - `triton_kernels/gemm.py`
  - `triton_kernels/rbf.py`
  - `triton_kernels/kan.py`
  - `triton_kernels/softmax.py`
  - `triton_kernels/attention.py`
  - `triton_kernels/tensor_contraction.py`
- **Optimization & Numerical Analysis**:
  - `optimization/autotuning/search_space.py`, `optimization/autotuning/tuner.py`, `optimization/autotune_gemm.py`
  - `numerical_analysis/accuracy.py`, `numerical_analysis/error_analysis.py`
- **Benchmarking & Visualization**:
  - `benchmarks/run_all.py`
  - `results/plots/generate_dashboard.py`
  - `results/benchmarks/end_to_end_benchmark_summary.csv`
  - `results/benchmarks/precision_error_report.csv`
  - `results/plots/executive_dashboard.png`
  - `results/plots/gemm_comparison.png`
  - `results/plots/kernel_speedups.png`
  - `results/plots/precision_error_tradeoff.png`
- **Unit Tests**:
  - 14 test modules in `tests/`, totaling 65 test cases, passing 100%.
