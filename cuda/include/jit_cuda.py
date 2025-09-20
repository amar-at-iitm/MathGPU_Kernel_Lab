"""
Dynamic CUDA C++ JIT Compiler & Runtime Launcher using NVRTC and CUDA Driver API.
Enables compiling and executing native CUDA kernels directly on the active GPU
without requiring an external offline MSVC / nvcc toolchain.
"""

import os
import sys
import glob
import ctypes
from typing import Dict, Any, List, Optional, Tuple
import torch


class CudaJitEngine:
    """Manages compilation of CUDA C++ source code to PTX and execution on GPU."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CudaJitEngine, cls).__new__(cls)
            cls._instance._init_driver_and_nvrtc()
        return cls._instance

    def _init_driver_and_nvrtc(self):
        # 1. Load CUDA Driver API
        if sys.platform == "win32":
            self.cuda = ctypes.CDLL("nvcuda.dll")
        else:
            self.cuda = ctypes.CDLL("libcuda.so")

        # Initialize driver
        ret = self.cuda.cuInit(0)
        if ret != 0:
            raise RuntimeError(f"cuInit(0) failed with error code: {ret}")

        # Ensure PyTorch creates and binds the CUDA context on the current thread
        if torch.cuda.is_available():
            torch.cuda.init()
            _ = torch.empty(1, device="cuda")

        # 2. Locate and load NVRTC
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        nvrtc_dlls = glob.glob(os.path.join(torch_lib, "*nvrtc64*")) or glob.glob(os.path.join(torch_lib, "*nvrtc*"))
        if not nvrtc_dlls:
            raise RuntimeError(f"Could not locate NVRTC library in PyTorch lib dir: {torch_lib}")

        # Prefer versioned nvrtc64_*.dll
        nvrtc_path = sorted(nvrtc_dlls, reverse=True)[0]
        self.nvrtc = ctypes.CDLL(nvrtc_path)
        self._compiled_cache: Dict[str, Tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}

    def compile_and_load(self,
                         cuda_source: str,
                         kernel_name: str,
                         include_dirs: Optional[List[str]] = None) -> ctypes.c_void_p:
        """
        Compile CUDA source to PTX via NVRTC and load function handle into the GPU driver.
        """
        cache_key = f"{kernel_name}_{hash(cuda_source)}"
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        # Determine architecture
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch = f"compute_{major}{minor}"
        else:
            arch = "compute_80"

        # Prepare NVRTC options
        opts = [f"--gpu-architecture={arch}".encode("utf-8"), b"--use_fast_math", b"-default-device"]
        if include_dirs:
            for inc in include_dirs:
                opts.append(f"-I{inc}".encode("utf-8"))

        opts_arr = (ctypes.c_char_p * len(opts))(*opts)

        # Create program
        prog = ctypes.c_void_p()
        src_bytes = cuda_source.encode("utf-8")
        ret = self.nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            src_bytes,
            f"{kernel_name}.cu".encode("utf-8"),
            0, None, None
        )
        if ret != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed with error code: {ret}")

        # Compile
        compile_res = self.nvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
        if compile_res != 0:
            # Retrieve log
            log_size = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            self.nvrtc.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(f"NVRTC Compilation Error for {kernel_name}:\n{log_buf.value.decode('utf-8')}")

        # Get PTX
        ptx_size = ctypes.c_size_t()
        self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        self.nvrtc.nvrtcGetPTX(prog, ptx_buf)

        # Destroy program
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        # Load PTX into CUDA Driver
        cu_module = ctypes.c_void_p()
        load_ret = self.cuda.cuModuleLoadData(ctypes.byref(cu_module), ptx_buf.value)
        if load_ret != 0:
            raise RuntimeError(f"cuModuleLoadData failed with error code: {load_ret}")

        # Get function pointer
        cu_func = ctypes.c_void_p()
        func_name_bytes = kernel_name.encode("utf-8")
        func_ret = self.cuda.cuModuleGetFunction(ctypes.byref(cu_func), cu_module, func_name_bytes)
        if func_ret != 0:
            raise RuntimeError(f"cuModuleGetFunction failed for '{kernel_name}' with code: {func_ret}")

        self._compiled_cache[cache_key] = cu_func
        return cu_func

    def launch(self,
               cu_func: ctypes.c_void_p,
               grid: Tuple[int, int, int],
               block: Tuple[int, int, int],
               args: List[Any],
               shared_mem_bytes: int = 0,
               stream: Optional[torch.cuda.Stream] = None) -> None:
        """Launch the loaded CUDA kernel function with given grid/block configuration."""
        stream_ptr = ctypes.c_void_p(
            stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream
        )

        c_values = []
        c_arg_refs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                c_val = ctypes.c_uint64(arg.data_ptr())
            elif isinstance(arg, int):
                c_val = ctypes.c_int(arg)
            elif isinstance(arg, float):
                c_val = ctypes.c_float(arg)
            else:
                raise TypeError(f"Unsupported kernel argument type: {type(arg)}")
            c_values.append(c_val)
            c_arg_refs.append(ctypes.cast(ctypes.byref(c_val), ctypes.c_void_p))

        c_args_array = (ctypes.c_void_p * len(c_arg_refs))(*c_arg_refs)

        gx, gy, gz = grid
        bx, by, bz = block

        ret = self.cuda.cuLaunchKernel(
            cu_func,
            gx, gy, gz,
            bx, by, bz,
            shared_mem_bytes,
            stream_ptr,
            c_args_array,
            None
        )
        if ret != 0:
            raise RuntimeError(f"cuLaunchKernel failed with code: {ret}")


# Global singleton engine
_jit_engine = None

def get_cuda_jit_engine() -> CudaJitEngine:
    global _jit_engine
    if _jit_engine is None:
        _jit_engine = CudaJitEngine()
    return _jit_engine
