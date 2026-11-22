# cuda-tiled-gemm

[![CUDA Build Check](https://github.com/Khalifax98/cuda-tiled-gemm/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/Khalifax98/cuda-tiled-gemm/actions/workflows/cuda-build.yml)

An iterative, high-performance **FP32 General Matrix Multiplication (SGEMM)** engine written in **CUDA C++17**, progressing systematically from a naive global-memory baseline to shared-memory tiling and 2D register coarsening, benchmarked against **NVIDIA cuBLAS**.

---

## Optimization Progression

1. **Stage 1 — Naive Global Memory (`src/naive_gemm.cu`)**  
   Each thread computes a single output element $C_{i,j}$ by reading entire rows of $A$ and columns of $B$ directly from global DRAM. Heavily memory-bandwidth bound.
2. **Stage 2 — Shared Memory Tiling (`src/tiled_gemm.cu`)**  
   Partitions matrices into $32 \times 32$ sub-blocks loaded cooperatively into fast on-chip `__shared__` memory, reducing global DRAM traffic by $32\times$.
3. **Stage 3 — Bank-Padded Shared Memory (`src/tiled_gemm.cu`)**  
   Evaluates `+1` column padding (`__shared__ float s_A[32][33]`) to analyze shared-memory bank conflict behavior vs. addressing overhead on Turing (SM 7.5).
4. **Stage 4 — 2D Register Coarsening (`src/coarsened_gemm.cu`)**  
   Each thread block computes a $128 \times 128$ output tile using 256 threads, where each thread accumulates an $8 \times 8$ micro-tile in registers via outer products—maximizing arithmetic intensity and achieving **up to 82.3% of cuBLAS throughput**.

---

## Tesla T4 (SM 7.5) Benchmark Results

Benchmarked using `cudaEvent_t` (20 timed runs after 5 warmup iterations, excluding host-device `cudaMemcpy` overhead) and verified against `cublasSgemm` via L2 relative error:

### Matrix Size: $1024 \times 1024 \times 1024$
| Kernel Stage | Time (ms) | Throughput (GFLOPS) | % of cuBLAS | Relative L2 Error |
| :--- | :---: | :---: | :---: | :---: |
| **0. cuBLAS (Reference)** | 0.637 | 3,373.3 | 100.0% | `0.00e+00` |
| **1. Naive Global Memory** | 5.038 | 426.2 | 12.6% | `6.01e-07` |
| **2. Shared Memory Tiled** | 2.504 | 857.7 | 25.4% | `6.01e-07` |
| **3. Tiled + Bank Padded** | 3.206 | 669.9 | 19.9% | `6.01e-07` |
| **4. 2D Register Coarsened** | **0.773** | **2,776.7** | **82.3%** | `6.01e-07` |

### Matrix Size: $2048 \times 2048 \times 2048$
| Kernel Stage | Time (ms) | Throughput (GFLOPS) | % of cuBLAS | Relative L2 Error |
| :--- | :---: | :---: | :---: | :---: |
| **0. cuBLAS (Reference)** | 3.792 | 4,530.2 | 100.0% | `0.00e+00` |
| **1. Naive Global Memory** | 28.914 | 594.2 | 13.1% | `7.70e-07` |
| **2. Shared Memory Tiled** | 19.309 | 889.7 | 19.6% | `7.70e-07` |
| **3. Tiled + Bank Padded** | 24.003 | 715.7 | 15.8% | `7.70e-07` |
| **4. 2D Register Coarsened** | **5.397** | **3,183.2** | **70.3%** | `7.70e-07` |

### Matrix Size: $4096 \times 4096 \times 4096$
| Kernel Stage | Time (ms) | Throughput (GFLOPS) | % of cuBLAS | Relative L2 Error |
| :--- | :---: | :---: | :---: | :---: |
| **0. cuBLAS (Reference)** | 33.589 | 4,091.8 | 100.0% | `0.00e+00` |
| **1. Naive Global Memory** | 234.219 | 586.8 | 14.3% | `0.00e+00` |
| **2. Shared Memory Tiled** | 154.637 | 888.8 | 21.7% | `0.00e+00` |
| **3. Tiled + Bank Padded** | 196.472 | 699.5 | 17.1% | `0.00e+00` |
| **4. 2D Register Coarsened** | **45.978** | **2,989.2** | **73.1%** | `0.00e+00` |

> **Key Profiling Takeaway:** On SM 7.5, Stage 2's inner loop (`s_A[ty][k] * s_B[k][tx]`) broadcasts `s_A[ty][k]` across a warp and reads consecutive banks along `tx` in `s_B[k][tx]`, meaning it is already bank-conflict-free. Adding `+1` padding in Stage 3 adds address-calculation overhead without eliminating conflicts, whereas Stage 4's 2D register tiling (`8x8` per thread) cuts shared-memory load instructions by $8\times$ and delivers a **5.1x–6.5x speedup** over the naive baseline.

---

## Build & Run

### Requirements
- NVIDIA CUDA Toolkit $\ge 11.8$ (`nvcc`, `cuBLAS`)
- CMake $\ge 3.18$ & C++17 compiler

### Compilation & Execution
```bash
git clone [https://github.com/Khalifax98/cuda-tiled-gemm.git](https://github.com/Khalifax98/cuda-tiled-gemm.git)
cd cuda-tiled-gemm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/gemm_benchmark
