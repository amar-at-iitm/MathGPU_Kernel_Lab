#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// Safe CUDA error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s (%s)\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err),              \
                    cudaGetErrorName(err));                                   \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                    cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                               \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA kernel launch error at %s:%d - %s (%s)\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err),              \
                    cudaGetErrorName(err));                                   \
            throw std::runtime_error(std::string("CUDA launch error: ") +     \
                                    cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

namespace mathgpu {

constexpr int WARP_SIZE = 32;

// Ceiling division helper
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// CUDA Event-based microsecond precision timer for host-side C++
struct GpuTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&start_event));
        CHECK_CUDA(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(start_event, stream));
    }

    float stop(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(stop_event, stream));
        CHECK_CUDA(cudaEventSynchronize(stop_event));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms; // returns milliseconds
    }
};

} // namespace mathgpu
