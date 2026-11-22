Cloning into 'cuda-tiled-gemm'...
remote: Enumerating objects: 40, done.
remote: Counting objects: 100% (40/40), done.
remote: Compressing objects: 100% (34/34), done.
remote: Total 40 (delta 13), reused 0 (delta 0), pack-reused 0 (from 0)
Receiving objects: 100% (40/40), 18.48 KiB | 3.70 MiB/s, done.
Resolving deltas: 100% (13/13), done.
/content/cuda-tiled-gemm
-- The CXX compiler identification is GNU 13.3.0
-- The CUDA compiler identification is NVIDIA 13.0.88 with host compiler GNU 13.3.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include;/usr/local/cuda/targets/x86_64-linux/include/cccl (found version "13.0.88")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Configuring done (4.9s)
-- Generating done (0.0s)
-- Build files have been written to: /content/cuda-tiled-gemm/build
[ 20%] Building CUDA object CMakeFiles/gemm_benchmark.dir/src/naive_gemm.cu.o
[ 40%] Building CUDA object CMakeFiles/gemm_benchmark.dir/src/main.cu.o
[ 60%] Building CUDA object CMakeFiles/gemm_benchmark.dir/src/tiled_gemm.cu.o
[ 80%] Building CUDA object CMakeFiles/gemm_benchmark.dir/src/coarsened_gemm.cu.o
[100%] Linking CUDA executable gemm_benchmark
[100%] Built target gemm_benchmark
=== CUDA Tiled GEMM Benchmark Suite ===
Device: Tesla T4 (SM 7.5)

--------------------------------------------------------------------------------
Matrix Size: 1024 x 1024 x 1024
Kernel Stage                   Time (ms)        GFLOPS      % cuBLAS    Rel L2 Err
--------------------------------------------------------------------------------
0. cuBLAS (Reference)              0.637        3373.3       100.0%      0.00e+00
1. Naive Global Memory             5.038         426.2         12.6%      6.01e-07
2. Shared Memory Tiled             2.504         857.7         25.4%      6.01e-07
3. Tiled + Bank Padded             3.206         669.9         19.9%      6.01e-07
4. 2D Register Coarsened           0.773        2776.7         82.3%      6.01e-07

--------------------------------------------------------------------------------
Matrix Size: 2048 x 2048 x 2048
Kernel Stage                   Time (ms)        GFLOPS      % cuBLAS    Rel L2 Err
--------------------------------------------------------------------------------
0. cuBLAS (Reference)              3.792        4530.2       100.0%      0.00e+00
1. Naive Global Memory            28.914         594.2         13.1%      7.70e-07
2. Shared Memory Tiled            19.309         889.7         19.6%      7.70e-07
3. Tiled + Bank Padded            24.003         715.7         15.8%      7.70e-07
4. 2D Register Coarsened           5.397        3183.2         70.3%      7.70e-07

--------------------------------------------------------------------------------
Matrix Size: 4096 x 4096 x 4096
Kernel Stage                   Time (ms)        GFLOPS      % cuBLAS    Rel L2 Err
--------------------------------------------------------------------------------
0. cuBLAS (Reference)             33.589        4091.8       100.0%      0.00e+00
1. Naive Global Memory           234.219         586.8         14.3%      0.00e+00
2. Shared Memory Tiled           154.637         888.8         21.7%      0.00e+00
3. Tiled + Bank Padded           196.472         699.5         17.1%      0.00e+00
4. 2D Register Coarsened          45.978        2989.2         73.1%      0.00e+00

