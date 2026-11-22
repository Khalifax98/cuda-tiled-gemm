#include "gemm_kernels.cuh"

// Block tile dimensions
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Thread micro-tile dimensions (each thread computes an 8x8 block of C)
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void coarsened_sgemm_kernel(int M, int N, int K, float alpha,
                                       const float* __restrict__ A,
                                       const float* __restrict__ B, float beta,
                                       float* __restrict__ C) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = (BM / TM) * (BN / TN); // 256

    __shared__ float s_A[BM][BK + 1];
    __shared__ float s_B[BK][BN + 1];

    float thread_results[TM][TN] = {0.0f};
    float reg_A[TM] = {0.0f};
    float reg_B[TN] = {0.0f};

    const int load_A_row = tid / BK;
    const int load_A_col = tid % BK;
    const int stride_A = num_threads / BK;

    const int load_B_row = tid / BN;
    const int load_B_col = tid % BN;
    const int stride_B = num_threads / BN;

    const int block_row_start = blockIdx.y * BM;
    const int block_col_start = blockIdx.x * BN;

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {
        #pragma unroll
        for (int offset = 0; offset < BM; offset += stride_A) {
            int global_r = block_row_start + load_A_row + offset;
            int global_c = bk_idx + load_A_col;
            if (global_r < M && global_c < K) {
                s_A[load_A_row + offset][load_A_col] = A[global_r * K + global_c];
            } else {
                s_A[load_A_row + offset][load_A_col] = 0.0f;
            }
        }

        #pragma unroll
        for (int offset = 0; offset < BK; offset += stride_B) {
            int global_r = bk_idx + load_B_row + offset;
            int global_c = block_col_start + load_B_col;
            if (global_r < K && global_c < N) {
                s_B[load_B_row + offset][load_B_col] = B[global_r * N + global_c];
            } else {
                s_B[load_B_row + offset][load_B_col] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = s_A[threadIdx.y * TM + i][dot_idx];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[dot_idx][threadIdx.x * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = block_row_start + threadIdx.y * TM + i;
        if (global_r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int global_c = block_col_start + threadIdx.x * TN + j;
                if (global_c < N) {
                    int idx = global_r * N + global_c;
                    C[idx] = alpha * thread_results[i][j] + beta * C[idx];
                }
            }
        }
    }
}

void launch_coarsened_gemm(int M, int N, int K, float alpha, const float* d_A,
                           const float* d_B, float beta, float* d_C) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    coarsened_sgemm_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
