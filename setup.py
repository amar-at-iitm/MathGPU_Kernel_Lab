import os
import sys
from setuptools import setup, find_packages

# Determine if CUDA toolkit is available for building C++/CUDA extensions
build_cuda = False
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
    build_cuda = torch.cuda.is_available()
except ImportError:
    pass

ext_modules = []

# Optional CUDA extensions configuration if CUDA_HOME or nvcc is available
if build_cuda and os.environ.get("CUDA_HOME") is not None:
    extra_compile_args = {
        'cxx': ['-O3'],
        'nvcc': ['-O3', '--use_fast_math', '-lineinfo']
    }
    if sys.platform == 'win32':
        extra_compile_args['cxx'] = ['/O2', '/std:c++17']
        extra_compile_args['nvcc'] = ['-O3', '--use_fast_math']

    sources = []
    if os.path.exists("cuda/naive/gemm_naive.cu"):
        sources.append("cuda/naive/gemm_naive.cu")
    if os.path.exists("cuda/tiled/gemm_tiled.cu"):
        sources.append("cuda/tiled/gemm_tiled.cu")

    if sources:
        ext_modules.append(
            CUDAExtension(
                name="mathgpu_cuda_kernels",
                sources=sources,
                include_dirs=[os.path.abspath("cuda/include")],
                extra_compile_args=extra_compile_args,
            )
        )

cmdclass = {'build_ext': BuildExtension} if ext_modules else {}

setup(
    name="mathgpu_kernel_lab",
    version="0.1.0",
    description="Mathematical Optimization of GPU Tensor Kernels",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
