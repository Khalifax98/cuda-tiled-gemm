#include "gemm_kernels.cuh"

constexpr int TILE_SIZE = 32;

// Stage 2: Shared Memory Tiling
__global__ void tiled_sgemm_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B, float beta,
                                   float* __restrict__ C) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col_A = t * TILE_SIZE + tx;
        int tiled_row_B = t * TILE_SIZE + ty;

        if (row < M && tiled_col_A < K) {
            s_A[ty][tx] = A[row * K + tiled_col_A];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        if (tiled_row_B < K && col < N) {
            s_B[ty][tx] = B[tiled_row_B * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Stage 3: Shared Memory Tiling with +1 Padding to prevent bank conflicts
__global__ void tiled_padded_sgemm_kernel(int M, int N, int K, float alpha,
                                          const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float beta, float* __restrict__ C) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col_A = t * TILE_SIZE + tx;
        int tiled_row_B = t * TILE_SIZE + ty;

        s_A[ty][tx] = (row < M && tiled_col_A < K) ? A[row * K + tiled_col_A] : 0.0f;
        s_B[ty][tx] = (tiled_row_B < K && col < N) ? B[tiled_row_B * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void launch_tiled_gemm(int M, int N, int K, float alpha, const float* d_A,
                       const float* d_B, float beta, float* d_C) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_sgemm_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}

void launch_tiled_padded_gemm(int M, int N, int K, float alpha,
                              const float* d_A, const float* d_B, float beta,
                              float* d_C) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_padded_sgemm_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
