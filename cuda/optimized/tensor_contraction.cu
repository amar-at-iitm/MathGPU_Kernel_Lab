// CUDA 4D Tensor Contraction Kernel
// Operation: C_{ijkl} = \sum_{a,b} A_{ijab} * B_{abkl}
// Implemented with shared-memory tiling over the contracted dimensions (a, b)

#define TILE_DIM 16

extern "C" __global__ void tensor_contraction_4d_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int I, int J, int K_dim, int L,
    int A_dim, int B_dim
) {
    int M = I * J;
    int N = K_dim * L;
    int K = A_dim * B_dim;

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // index in M
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // index in N

    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    float acc = 0.0f;

    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_k_A = t * TILE_DIM + threadIdx.x;
        if (row < M && tiled_k_A < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + tiled_k_A];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int tiled_k_B = t * TILE_DIM + threadIdx.y;
        if (tiled_k_B < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[tiled_k_B * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
