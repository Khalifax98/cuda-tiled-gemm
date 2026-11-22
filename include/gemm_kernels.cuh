#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS Error at %s:%d - code %d\n", __FILE__,     \
                    __LINE__, static_cast<int>(status));                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void launch_naive_gemm(int M, int N, int K, float alpha, const float* d_A,
                       const float* d_B, float beta, float* d_C);

void launch_tiled_gemm(int M, int N, int K, float alpha, const float* d_A,
                       const float* d_B, float beta, float* d_C);

void launch_tiled_padded_gemm(int M, int N, int K, float alpha,
                              const float* d_A, const float* d_B, float beta,
                              float* d_C);

void launch_coarsened_gemm(int M, int N, int K, float alpha, const float* d_A,
                           const float* d_B, float beta, float* d_C);
