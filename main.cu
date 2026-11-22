#include "gemm_kernels.cuh"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

float compute_relative_l2_error(const std::vector<float>& ref,
                                const std::vector<float>& target) {
    double diff_sq_sum = 0.0;
    double ref_sq_sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = static_cast<double>(ref[i]) - static_cast<double>(target[i]);
        diff_sq_sum += diff * diff;
        ref_sq_sum += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return static_cast<float>(std::sqrt(diff_sq_sum / (ref_sq_sum + 1e-12)));
}

float benchmark_kernel(const std::function<void()>& launch_fn, int warmup_runs,
                       int timed_runs) {
    for (int i = 0; i < warmup_runs; ++i) {
        launch_fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; ++i) {
        launch_fn();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(timed_runs);
}

int main(int argc, char** argv) {
    int device_id = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    std::cout << "=== CUDA Tiled GEMM Benchmark Suite ===\n";
    std::cout << "Device: " << props.name
              << " (SM " << props.major << "." << props.minor << ")\n\n";

    std::vector<int> sizes = {1024, 2048, 4096};
    if (argc > 1) {
        sizes = {std::stoi(argv[1])};
    }

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int warmup_runs = 5;
    const int timed_runs = 20;

    for (int N : sizes) {
        int M = N, K = N;
        size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
        double total_flops = 2.0 * static_cast<double>(M) * N * K;

        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << "Matrix Size: " << M << " x " << N << " x " << K << "\n";
        std::cout << std::left << std::setw(28) << "Kernel Stage"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(14) << "GFLOPS"
                  << std::setw(14) << "% cuBLAS"
                  << std::setw(14) << "Rel L2 Err" << "\n";
        std::cout << "--------------------------------------------------------------------------------\n";

        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        std::vector<float> h_C_ref(M * N, 0.0f);
        std::vector<float> h_C_out(M * N, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : h_A) v = dist(gen);
        for (auto& v : h_B) v = dist(gen);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        auto run_cublas = [&]() {
            CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, d_B, N, d_A, K, &beta,
                                     d_C, N));
        };

        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        float cublas_ms = benchmark_kernel(run_cublas, warmup_runs, timed_runs);
        double cublas_gflops = (total_flops / (cublas_ms * 1e-3)) / 1e9;
        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        std::cout << std::left << std::setw(28) << "0. cuBLAS (Reference)"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << cublas_ms
                  << std::setprecision(1) << std::setw(14) << cublas_gflops
                  << std::setw(13) << "100.0%"
                  << std::scientific << std::setprecision(2) << std::setw(14) << 0.0f
                  << "\n";

        struct Stage {
            std::string name;
            std::function<void()> fn;
        };

        std::vector<Stage> stages = {
            {"1. Naive Global Memory",
             [&]() { launch_naive_gemm(M, N, K, alpha, d_A, d_B, beta, d_C); }},
            {"2. Shared Memory Tiled",
             [&]() { launch_tiled_gemm(M, N, K, alpha, d_A, d_B, beta, d_C); }},
            {"3. Tiled + Bank Padded",
             [&]() { launch_tiled_padded_gemm(M, N, K, alpha, d_A, d_B, beta, d_C); }},
            {"4. 2D Register Coarsened",
             [&]() { launch_coarsened_gemm(M, N, K, alpha, d_A, d_B, beta, d_C); }}
        };

        for (const auto& stage : stages) {
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));
            stage.fn();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_out.data(), d_C, bytes, cudaMemcpyDeviceToHost));
            float rel_err = compute_relative_l2_error(h_C_ref, h_C_out);

            float avg_ms = benchmark_kernel(stage.fn, warmup_runs, timed_runs);
            double gflops = (total_flops / (avg_ms * 1e-3)) / 1e9;
            double pct_cublas = (gflops / cublas_gflops) * 100.0;

            std::cout << std::left << std::setw(28) << stage.name
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(12) << avg_ms
                      << std::setprecision(1) << std::setw(14) << gflops
                      << std::setw(13) << pct_cublas << "%"
                      << std::scientific << std::setprecision(2) << std::setw(14) << rel_err
                      << "\n";
        }

        std::cout << "\n";
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
