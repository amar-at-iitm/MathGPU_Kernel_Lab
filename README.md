# **`MathGPU_Kernel_Lab`**

### Mathematical Optimization of GPU Tensor Kernels for Scientific and AI Computing

MathGPU_Kernel_Lab is a research-oriented GPU computing project focused on designing, implementing, profiling, and optimizing mathematical and tensor operations on modern NVIDIA GPUs.

The project investigates how mathematical structure, numerical methods, memory hierarchy, parallel execution, and GPU-specific hardware features can be jointly exploited to build high-performance computational kernels.

The implementations span:

- CUDA C++
- Triton
- CUTLASS
- PyTorch
- Tensor Cores
- Mixed-precision arithmetic
- GPU profiling and performance analysis
- Automatic kernel tuning

The central objective is:

> **Translate mathematically structured computations into efficient GPU kernels and systematically optimize their execution on modern GPU architectures.**


**Instead of treating GPU programming as a purely low-level programming problem, the project formulates kernel optimization as a mathematical optimization problem:**

$$\theta^* = \arg\min_{\theta \in \Theta} T(\theta) $$

where:

- $T(\theta)$ is the measured execution time,
- $\theta$ represents kernel configuration parameters,
- $\Theta$ is the feasible configuration space.

The project therefore combines:

$$\boxed{\text{Mathematics}+\text{Numerical Computing}+\text{GPU Architecture}+\text{Tensor Programming}}$$

---

# Project Objectives

The main objectives are:

1. Implement fundamental mathematical operations directly using CUDA.
2. Develop equivalent kernels using Triton.
3. Investigate GPU memory hierarchy and parallel execution.
4. Exploit shared memory, registers, and warp-level parallelism.
5. Explore Tensor Core acceleration and mixed precision.
6. Compare custom implementations with PyTorch and vendor libraries.
7. Profile kernels using NVIDIA Nsight tools.
8. Analyze numerical accuracy and floating-point error.
9. Develop an automated kernel-tuning framework.
10. Study the relationship between mathematical structure and GPU performance.

---

# Operations

The project progressively implements and optimizes several computational kernels.

## 1. Matrix Multiplication

The fundamental operation is

$$
C = AB.
$$

Implementations include:

- Naive CUDA GEMM
- Tiled CUDA GEMM
- Shared-memory GEMM
- Tensor Core GEMM
- Triton GEMM
- CUTLASS GEMM
- PyTorch/cuBLAS baseline

Performance is evaluated using:

$$\text{FLOPS}=\frac{2MNK}{T}.$$

---

## 2. Radial Basis Function Kernel

The Gaussian RBF is defined as

$$K(x_i,x_j)=\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right).$$

The project implements GPU kernels for:

- pairwise distance computation,
- Gaussian evaluation,
- batched RBF computation,
- mixed-precision RBF computation.

The objective is to investigate whether mathematical structure can be exploited to reduce memory traffic and improve GPU utilization.

---

## 3. GPU-Accelerated KAN/RBF Layer

A mathematical basis-function layer is considered in the form

$$y_j=\sum_i \phi_{ij}(x_i),$$

where the basis function can be represented using Gaussian RBFs:

$$\phi_{ij}(x)=\sum_{k=1}^{K}c_{ijk}\exp\left(-\frac{(x-\mu_k)^2}{2\sigma_k^2}\right).$$

The project investigates:

- vectorized implementations,
- CUDA kernels,
- Triton kernels,
- memory-efficient evaluation,
- batched execution,
- mixed precision,
- kernel fusion.

This component connects GPU computing with mathematical machine-learning architectures.

---

# 4. Tensor Contractions

Tensor contractions are common in scientific computing, physics, numerical linear algebra, and machine learning.

A representative operation is

$$C_{ijkl}=\sum_{a,b}A_{ijab}B_{abkl}.$$

The implementation investigates:

- tensor layouts,
- indexing strategies,
- memory coalescing,
- tiling,
- shared memory,
- register utilization,
- and parallel decomposition.

---

# 5. Softmax

For an input vector $x$,

$$\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}.$$

The CUDA implementation focuses on efficient:

- reduction,
- maximum computation,
- exponential evaluation,
- normalization,
- and memory access.

Numerical stability is maintained using

$$\text{softmax}(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},$$

where

$$m=\max_j x_j.$$

---

# 6. Attention

The project implements the computational components of attention:

$$S = QK^T$$

$$P = \text{softmax}(S)$$

$$O = PV.$$

Different implementations are compared:

```text
PyTorch
   │
   ├── Baseline Attention
   │
CUDA
   │
   ├── Naive CUDA
   ├── Tiled CUDA
   └── Fused CUDA
   │
Triton
   │
   └── Optimized Triton
````

The project studies memory traffic, tiling, fusion, and GPU utilization for attention workloads.

---

# GPU Programming Layers

Each mathematical operation is implemented at multiple abstraction levels.

```text
                    Mathematical Operation
                             │
                ┌────────────┼────────────┐
                ↓            ↓            ↓
             PyTorch       Triton       CUDA
                │            │            │
                └────────────┼────────────┘
                             ↓
                          CUTLASS
                             │
                             ↓
                      Tensor Cores
                             │
                             ↓
                    Hardware Profiling
```

This enables systematic comparison between high-level frameworks and custom GPU kernels.

---

# CUDA Optimization Techniques

The CUDA implementations progressively introduce hardware-aware optimizations.

## Memory Coalescing

Threads are mapped so that adjacent threads access adjacent memory locations.

## Shared Memory

Frequently reused data is placed in low-latency shared memory.

## Tiling

Large matrix/tensor operations are decomposed into smaller tiles.

## Register Blocking

Frequently accessed values are kept in registers to reduce memory traffic.

## Warp-Level Parallelism

Operations are mapped to GPU warps to exploit SIMD-style execution.

## Kernel Fusion

Multiple computational stages are combined into a single kernel to reduce:

* global-memory traffic,
* kernel launch overhead,
* intermediate tensor storage.

---

# Tensor Core Acceleration

Tensor Cores are investigated for matrix and tensor operations using mixed precision.

Supported numerical formats include:

* FP32
* TF32
* FP16
* BF16

The project compares:

$$
\text{FP32}
\rightarrow
\text{TF32}
\rightarrow
\text{FP16/BF16}
$$

in terms of:

* execution time,
* throughput,
* numerical accuracy,
* memory consumption.

Accuracy is evaluated against high-precision reference implementations.

---

# Performance Metrics

Each implementation is evaluated using multiple metrics.

### Latency

$$T_{\text{kernel}}$$

### Throughput

$$\text{Throughput}=\frac{\text{Operations}}{T}.$$

### Floating Point Performance

$$\text{FLOPS}=\frac{\text{Number of floating-point operations}}{\text{Execution time}}.$$

### Speedup

$$S=\frac{T_{\text{baseline}}}{T_{\text{optimized}}}.$$

### Memory Bandwidth

$$BW=\frac{\text{Bytes transferred}}{T}.$$

### Numerical Error

For reference result $y$ and GPU result $\hat y$,

$$E_{\text{relative}}=\frac{\|y-\hat y\|_2}{\|y\|_2}.$$

---

# GPU Profiling

Performance bottlenecks are investigated using NVIDIA profiling tools.

## NVIDIA Nsight Systems

Used for:

* kernel execution timelines,
* CPU-GPU synchronization,
* kernel launch overhead,
* data transfers,
* overall application behavior.

## NVIDIA Nsight Compute

Used for:

* occupancy,
* memory throughput,
* SM utilization,
* register usage,
* shared-memory utilization,
* warp efficiency,
* instruction statistics,
* Tensor Core utilization.

---

# Roofline Analysis

The project uses the roofline model to understand whether an operation is:

* compute-bound, or
* memory-bound.

Arithmetic intensity is defined as

$$AI=\frac{\text{FLOPs}}{\text{Bytes transferred}}.$$

The achievable performance is bounded approximately by

$$P\leq\min(P_{\text{peak}},AI \times BW_{\text{peak}}).$$

This allows mathematical operations to be analyzed from a hardware-performance perspective.

---

# Automatic Kernel Optimization

One of the main research components is automatic kernel tuning.

GPU kernel performance depends on parameters such as:

$$\theta =(B_M,B_N,B_K,W,S,P)$$

where parameters may represent:

* tile dimensions,
* number of warps,
* pipeline stages,
* threads,
* shared-memory allocation.

The optimization problem becomes:

$$\theta^*=\arg\min_{\theta}T(\theta).$$

The project explores:

* grid search,
* random search,
* Bayesian optimization,
* evolutionary optimization,
* heuristic search.

The tuner evaluates candidate configurations on the target GPU and identifies configurations that minimize measured latency.

---

# Experimental Methodology

Each experiment follows the same workflow:

```text
Mathematical Formulation
          ↓
Reference CPU Implementation
          ↓
PyTorch Baseline
          ↓
Naive CUDA Kernel
          ↓
Optimized CUDA Kernel
          ↓
Triton Implementation
          ↓
Tensor Core Implementation
          ↓
Profiling
          ↓
Automatic Tuning
          ↓
Numerical Validation
          ↓
Benchmark
```

All implementations are evaluated using identical input sizes and numerical tolerances.

---

# Repository Structure

```text
MathGPU_Kernel_Lab/
│
├── benchmarks/
│   ├── gemm/
│   ├── rbf/
│   ├── kan/
│   ├── attention/
│   └── tensor_contraction/
│
├── cuda/
│   ├── naive/
│   ├── tiled/
│   ├── shared_memory/
│   ├── tensor_core/
│   └── optimized/
│
├── triton/
│   ├── gemm.py
│   ├── rbf.py
│   ├── kan.py
│   ├── softmax.py
│   └── attention.py
│
├── cutlass/
│   └── ...
│
├── pytorch/
│   └── baselines/
│
├── numerical_analysis/
│   ├── accuracy.py
│   └── error_analysis.py
│
├── optimization/
│   ├── grid_search/
│   ├── bayesian/
│   ├── evolutionary/
│   └── autotuning/
│
├── profiling/
│   ├── nsight_systems/
│   ├── nsight_compute/
│   └── roofline/
│
├── tests/
│
├── scripts/
│
├── results/
│   ├── benchmarks/
│   ├── profiles/
│   └── plots/
│
├── docs/
│
├── CMakeLists.txt
├── requirements.txt
├── environment.yml
└── README.md
```

---

# Hardware Requirements

The project is designed primarily for NVIDIA GPUs supporting CUDA.

Recommended:

* NVIDIA RTX GPU
* NVIDIA A-series GPU
* NVIDIA H100/H200
* NVIDIA L40/L40S
* NVIDIA A100

Tensor Core experiments require compatible GPU hardware.

---

# Software Requirements

Recommended environment:

```text
Ubuntu 22.04+
CUDA 12.x+
Python 3.10+
C++17+
CMake 3.20+
PyTorch
Triton
CUTLASS
NumPy
Matplotlib
Pandas
```

Optional:

```text
NVIDIA Nsight Systems
NVIDIA Nsight Compute
CUDA Profiler
Jupyter
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<username>/MathGPU_Kernel_Lab.git
cd MathGPU_Kernel_Lab
```

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify CUDA:

```bash
nvidia-smi
```

Verify the CUDA compiler:

```bash
nvcc --version
```

---

# Building CUDA Kernels

Create a build directory:

```bash
mkdir build
cd build
```

Configure:

```bash
cmake ..
```

Build:

```bash
make -j$(nproc)
```

---

# Running Benchmarks

Example:

```bash
python benchmarks/gemm/benchmark.py
```

RBF benchmark:

```bash
python benchmarks/rbf/benchmark.py
```

KAN benchmark:

```bash
python benchmarks/kan/benchmark.py
```

Attention benchmark:

```bash
python benchmarks/attention/benchmark.py
```

---

# Example Benchmark

A typical benchmark compares:

```text
                 Latency
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
     PyTorch     CUDA       Triton
        │          │          │
        └──────────┼──────────┘
                   ↓
              Optimized
                CUDA
                   │
                   ↓
              Tensor Core
```

The final results are reported using:

| Implementation   | Latency | Throughput | Speedup | Relative Error |
| ---------------- | ------: | ---------: | ------: | -------------: |
| PyTorch          |       - |          - |   1.00× |              - |
| Naive CUDA       |       - |          - |       - |              - |
| Tiled CUDA       |       - |          - |       - |              - |
| Triton           |       - |          - |       - |              - |
| Tensor Core      |       - |          - |       - |              - |
| Optimized Kernel |       - |          - |       - |              - |

Actual values are hardware-dependent.

---

# Numerical Validation

Performance improvements must not compromise correctness.

Each optimized implementation is compared against a reference implementation.

For example:

```python
torch.testing.assert_close(
    reference,
    optimized,
    rtol=1e-3,
    atol=1e-4
)
```

Different tolerances are used depending on numerical precision.

Special attention is given to:

* FP32
* TF32
* FP16
* BF16

and their accumulation behavior.



---

# Skills Demonstrated

### Mathematics

* Linear algebra
* Numerical analysis
* Optimization
* Tensor algebra
* RBF methods
* Mathematical machine learning

### GPU Computing

* CUDA C++
* GPU memory hierarchy
* Warp-level programming
* Shared memory
* Register optimization
* Tensor Cores
* Mixed precision

### AI Systems

* PyTorch
* Triton
* Tensor operations
* Attention
* KAN/RBF layers

### Performance Engineering

* Kernel benchmarking
* Nsight Systems
* Nsight Compute
* Roofline analysis
* Occupancy analysis
* Memory-bandwidth analysis
* Automatic kernel tuning

---

# Future Work

Potential extensions include:

* CUDA Graph optimization
* Multi-GPU execution
* Sparse tensor kernels
* GPU-accelerated PDE solvers
* GPU finite-element methods
* GPU spectral methods
* Automatic differentiation kernels
* Reinforcement-learning-based kernel optimization


---

# Research Questions

The project investigates several research questions.

### RQ1

How does mathematical structure affect GPU kernel performance?

### RQ2

When does an operation become memory-bound rather than compute-bound?

### RQ3

How do tile sizes affect GPU utilization?

### RQ4

How does mixed precision affect the accuracy-performance trade-off?

### RQ5

Can automatic optimization outperform manually selected kernel configurations?

### RQ6

How effectively can mathematical RBF/KAN operations exploit GPU parallelism?

### RQ7

How do CUDA and Triton compare for different classes of mathematical operations?

### RQ8

Which mathematical transformations enable effective kernel fusion?
